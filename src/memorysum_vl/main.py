from fastapi import FastAPI

from memorysum_vl.api.router import build_memory_router
from memorysum_vl.api.routes import health
from memorysum_vl.core.config import Settings, get_settings
from memorysum_vl.core.container import build_container
from memorysum_vl.core.errors import register_exception_handlers


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    app = FastAPI(title=resolved.app_name, version=resolved.app_version)
    app.state.container = build_container(resolved)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(build_memory_router(resolved.api_prefix))
    return app


app = create_app()
