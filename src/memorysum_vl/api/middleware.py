"""请求/响应日志中间件。

在 ASGI 层截获报文，因此无论后续是校验失败、鉴权失败、业务异常还是正常返回，
都能拿到完整的请求上下文与最终状态码。未预期异常在这里统一兜底成 500 错误信封，
避免同一份 traceback 被 uvicorn 再打印一次。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Final

from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from memorysum_vl.core.config import Settings
from memorysum_vl.core.logging import (
    get_logger,
    reset_request_id,
    sanitize_text,
    set_request_id,
)

logger = get_logger(__name__)

_BINARY_PREFIXES: Final[tuple[str, ...]] = (
    "image/",
    "multipart/",
    "application/octet-stream",
    "application/pdf",
)
_REQUEST_ID_HEADER: Final[bytes] = b"x-request-id"
_MAX_REQUEST_ID: Final[int] = 64
_ALLOWED_REQUEST_ID_CHARS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)
_INTERNAL_ERROR_BODY: Final[bytes] = json.dumps(
    {"error": {"code": "internal_error", "message": "internal server error"}}
).encode("utf-8")


class _BodyCapture:
    """保留正文；`limit <= 0` 表示全量保留，二进制载荷只记大小。"""

    __slots__ = ("_limit", "_chunks", "_stored", "_size", "_truncated", "_skip")

    def __init__(self, limit: int, *, skip: bool = False) -> None:
        self._limit = limit
        self._chunks: list[bytes] = []
        self._stored = 0
        self._size = 0
        self._truncated = False
        self._skip = skip

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._size += len(chunk)
        if self._skip:
            return
        if self._limit <= 0:  # 不限长度
            self._chunks.append(chunk)
            self._stored += len(chunk)
            return
        remaining = self._limit - self._stored
        if remaining <= 0:
            self._truncated = True
            return
        kept = chunk[:remaining]
        self._chunks.append(kept)
        self._stored += len(kept)
        if len(kept) < len(chunk):
            self._truncated = True

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    def render(self, *, limit: int) -> str:
        if self._skip:
            return f"<二进制内容 {format_bytes(self._size)}，未展开>"
        if self._size == 0:
            return ""
        text = b"".join(self._chunks).decode("utf-8", errors="replace")
        rendered = sanitize_text(text, limit=None if limit <= 0 else limit)
        if self._truncated:
            rendered = f"{rendered}…<共 {format_bytes(self._size)}>"
        return rendered


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self._body_limit = settings.log_request_body_limit
        self._response_limit = settings.log_response_body_limit
        self._quiet_paths = settings.quiet_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "-")
        path = scope.get("path", "")
        query = _query(scope)
        request_id = _incoming_request_id(scope) or uuid.uuid4().hex[:12]
        correlation = set_request_id(request_id)

        request_type = _header(scope, b"content-type")
        request_body = _BodyCapture(self._body_limit, skip=_is_binary(request_type))
        response_body = _BodyCapture(self._response_limit)

        status_code = 0
        response_type: str | None = None
        response_started = False
        failure: BaseException | None = None
        disconnected = False
        started = time.perf_counter()

        async def receive_wrapper() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                request_body.feed(message.get("body", b""))
            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_type, response_started
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_started = True
                headers = message.setdefault("headers", [])
                if not any(key.lower() == _REQUEST_ID_HEADER for key, _ in headers):
                    headers.append((_REQUEST_ID_HEADER, request_id.encode("latin-1")))
                response_type = _header_from_pairs(headers, b"content-type")
            elif message["type"] == "http.response.body":
                response_body.feed(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except ClientDisconnect:
            disconnected = True
        except Exception as exc:
            failure = exc
            if response_started:
                raise
            await self._send_internal_error(send_wrapper)
        finally:
            self._emit(
                method=method,
                path=path,
                query=query,
                scope=scope,
                request_type=request_type,
                request_body=request_body,
                response_type=response_type,
                response_body=response_body,
                status_code=status_code,
                started=started,
                failure=failure,
                disconnected=disconnected,
            )
            reset_request_id(correlation)

    @staticmethod
    async def _send_internal_error(send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_INTERNAL_ERROR_BODY)).encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _INTERNAL_ERROR_BODY})

    def _emit(
        self,
        *,
        method: str,
        path: str,
        query: str | None,
        scope: Scope,
        request_type: str | None,
        request_body: _BodyCapture,
        response_type: str | None,
        response_body: _BodyCapture,
        status_code: int,
        started: float,
        failure: BaseException | None,
        disconnected: bool,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        client_host, client_port = _client(scope)
        error_code = None
        if failure is not None:
            error_code = "internal_error"
        elif not disconnected:
            error_code = _extract_error_code(response_body)

        outcome = "client_disconnected" if disconnected else "exception" if failure else "ok"
        if failure is None and not disconnected and status_code >= 400:
            outcome = "rejected"

        lines = [
            _headline(
                method, path, query, status_code, error_code, elapsed_ms, client_host, client_port
            )
        ]
        user_agent = _header(scope, b"user-agent")
        if user_agent:
            lines[0] = f"{lines[0]} · {sanitize_text(user_agent, limit=120)}"
        if request_body.size:
            lines.append(_body_line("request", request_type, request_body, self._body_limit))
        if response_body.size:
            lines.append(_body_line("response", response_type, response_body, self._response_limit))
        if failure is not None:
            lines.append(
                f"  unhandled · {type(failure).__name__}: {sanitize_text(str(failure), limit=500)}"
            )
        if disconnected:
            lines.append("  client disconnected before the response was sent")

        level = _level(status_code, failure, disconnected, path, self._quiet_paths)
        logger.log(
            level,
            "\n".join(lines),
            extra={"outcome": outcome},
            exc_info=failure,
        )


def _level(
    status_code: int,
    failure: BaseException | None,
    disconnected: bool,
    path: str,
    quiet_paths: frozenset[str],
) -> int:
    if failure is not None or (status_code >= 500 and status_code != 501):
        return logging.ERROR
    # 501 表示能力尚未实现，属于已知状态，不必占用 ERROR
    if disconnected or status_code >= 400:
        return logging.WARNING
    if path in quiet_paths:
        return logging.DEBUG
    return logging.INFO


def _headline(
    method: str,
    path: str,
    query: str | None,
    status_code: int,
    error_code: str | None,
    elapsed_ms: float,
    client_host: str,
    client_port: int,
) -> str:
    target = f"{path}?{query}" if query else path
    if status_code:
        state = f"{status_code} {error_code}" if error_code else str(status_code)
    else:
        state = "无响应"
    return f"{method} {target} → {state} · {elapsed_ms:.1f} ms · {client_host}:{client_port}"


def _body_line(label: str, content_type: str | None, capture: _BodyCapture, limit: int) -> str:
    kind = content_type or "未知类型"
    rendered = capture.render(limit=limit)
    prefix = f"  {label:<9} · {format_bytes(capture.size)} · {kind}"
    return f"{prefix} · {rendered}" if rendered else prefix


def _extract_error_code(capture: _BodyCapture) -> str | None:
    if capture.is_empty:
        return None
    raw = capture.render(limit=4096)
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    error = parsed.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        return code if isinstance(code, str) else None
    return None


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _header_from_pairs(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _query(scope: Scope) -> str | None:
    raw = scope.get("query_string", b"")
    return raw.decode("latin-1") if raw else None


def _client(scope: Scope) -> tuple[str, int]:
    client = scope.get("client")
    if not client:
        return ("-", 0)
    return (str(client[0]), int(client[1]))


def _incoming_request_id(scope: Scope) -> str | None:
    raw = _header(scope, _REQUEST_ID_HEADER)
    if not raw:
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > _MAX_REQUEST_ID:
        return None
    if not set(candidate) <= _ALLOWED_REQUEST_ID_CHARS:
        return None
    return candidate


def _is_binary(content_type: str | None) -> bool:
    if not content_type:
        return False
    lowered = content_type.lower()
    return any(lowered.startswith(prefix) for prefix in _BINARY_PREFIXES)


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value / (1024 * 1024):.1f} MiB"
