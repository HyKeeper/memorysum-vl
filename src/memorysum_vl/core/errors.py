from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from memorysum_vl.schemas.errors import ErrorBody, ErrorResponse


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


async def _handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message, exc.headers)


async def _handle_not_implemented(_request: Request, exc: NotImplementedError) -> JSONResponse:
    return _error_response(501, "not_implemented", str(exc) or "capability is not implemented yet")


def _error_response(
    status_code: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)
