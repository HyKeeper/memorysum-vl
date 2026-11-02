"""应用装配入口。

用**工厂**方式启动，使 `import memorysum_vl.main` 没有副作用（不读 token、不连数据库），
测试与 CI 因此不需要任何凭据就能导入：

    uvicorn memorysum_vl.main:create_app --factory --app-dir src --port 8003

`memorysum_vl.main:app` 也仍然可用（见文件末尾的按需构造）。
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI

from memorysum_vl.api.middleware import RequestLoggingMiddleware
from memorysum_vl.api.router import build_memory_router
from memorysum_vl.api.routes import health
from memorysum_vl.core.config import Settings, get_settings
from memorysum_vl.core.container import Container, build_container
from memorysum_vl.core.errors import register_exception_handlers
from memorysum_vl.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved)

    active = container or build_container(resolved)
    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        lifespan=_lifespan(active),
    )
    app.state.container = active

    # 中间件必须在路由之前挂上，才能覆盖鉴权与校验阶段的失败
    app.add_middleware(RequestLoggingMiddleware, settings=resolved)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(build_memory_router(resolved.api_prefix))

    logger.info(
        "服务已装配",
        extra={
            "version": resolved.app_version,
            "track": resolved.track.value,
            "api_prefix": resolved.api_prefix,
            "store": type(active.memory_store).__name__,
            "db": str(resolved.db_path),
            "rrf_k": resolved.rrf_k,
            "rerank": resolved.rerank_enabled,
            "token_source": "file" if resolved.auth_token_file else "inline",
            "log_format": resolved.log_format,
            "log_file": str(resolved.log_file) if resolved.log_file else "仅 stderr",
            "request_body_limit": resolved.log_request_body_limit,
            "response_body_limit": resolved.log_response_body_limit,
        },
    )
    return app


def _lifespan(container: Container):
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        probe = asyncio.create_task(_probe(container))
        try:
            yield
        finally:
            probe.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await probe
            await container.aclose()

    return lifespan


async def _probe(container: Container) -> None:
    try:
        status = await container.probe()
    except Exception as exc:  # noqa: BLE001 - 探活失败只记日志，服务照常启动
        logger.warning("模型服务探活异常", extra={"error": type(exc).__name__})
        return
    missing = [name for name, model in status.items() if model is None]
    if missing:
        logger.warning(
            "有模型服务不可达，相关能力会降级",
            extra={"missing": ", ".join(missing), **status},
        )
        return
    logger.info("模型服务就绪", extra=status)


def __getattr__(name: str) -> FastAPI:
    """按需构造默认应用。

    `uvicorn memorysum_vl.main:app` 照旧可用，但只有真正取 `app` 时才会执行
    `create_app()`（此时才需要 token）。导入模块本身保持无副作用。
    """
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    application = create_app()
    globals()["app"] = application
    return application
