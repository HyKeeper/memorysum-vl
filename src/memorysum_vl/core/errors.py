"""异常统一映射为错误信封，并在这里把「为什么失败」完整写进日志。

真正的 traceback 由中间件兜底记录；本模块负责业务异常、校验失败与框架级
HTTP 错误的可读摘要，避免同一条异常在两处重复打印堆栈。
"""

from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from memorysum_vl.core.logging import get_logger, sanitize_value
from memorysum_vl.schemas.errors import ErrorBody, ErrorResponse

logger = get_logger(__name__)

_HTTP_CODE_NAMES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_failed",
    429: "too_many_requests",
}


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.headers = dict(headers or {})


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class UnsupportedContentError(AppError):
    status_code = 400
    code = "unsupported_content"


class UnsupportedTrackError(AppError):
    status_code = 400
    code = "unsupported_track"


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(NotImplementedError, _handle_not_implemented)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)


async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(
        f"业务异常 {exc.code}：{exc.message}",
        extra={
            "status": exc.status_code,
            "method": request.method,
            "path": request.url.path,
            "error_code": exc.code,
        },
    )
    return _error_response(exc.status_code, exc.code, exc.message, exc.headers)


async def _handle_not_implemented(request: Request, exc: NotImplementedError) -> JSONResponse:
    message = str(exc) or "capability is not implemented yet"
    logger.warning(
        f"能力未实现：{message}",
        extra={
            "status": 501,
            "method": request.method,
            "path": request.url.path,
            "error_code": "not_implemented",
        },
        exc_info=exc,
    )
    return _error_response(501, "not_implemented", message)


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = exc.errors()
    logger.warning(
        f"请求校验失败：{len(details)} 处问题",
        extra={
            "status": 422,
            "method": request.method,
            "path": request.url.path,
            "error_code": "validation_failed",
            "details": sanitize_value(_summarize(details), limit=300),
        },
    )
    return await request_validation_exception_handler(request, exc)


async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_CODE_NAMES.get(exc.status_code, f"http_{exc.status_code}")
    logger.warning(
        f"HTTP {exc.status_code}：{exc.detail}",
        extra={
            "status": exc.status_code,
            "method": request.method,
            "path": request.url.path,
            "error_code": code,
        },
    )
    return _error_response(exc.status_code, code, str(exc.detail), exc.headers)


def _summarize(details: list[dict]) -> list[str]:
    """把 pydantic 的校验错误压成一行人话，避免整段 input 灌进日志。"""
    summarized: list[str] = []
    for item in details:
        location = ".".join(str(part) for part in item.get("loc", ()))
        message = item.get("msg", "invalid value")
        summarized.append(f"{location or '<body>'}: {message}")
    return summarized


def _error_response(
    status_code: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)
