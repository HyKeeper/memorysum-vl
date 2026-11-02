"""人类可读的结构化日志：格式化、脱敏与请求关联。

每条记录以 `时间 级别 [请求ID] 模块 │` 开头，正文内的续行沿用同一前缀，
因此 `tail -f var/log/memorysum.log` 能看出层级，journald 也能逐行收下。

脱敏规则在写入前统一执行，任何 handler 都拿不到明文凭据或图片载荷。
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from contextvars import ContextVar, Token
from datetime import datetime
from typing import Any, Final

from memorysum_vl.core.config import Settings

REDACTED: Final[str] = "***"
NO_REQUEST_ID: Final[str] = "-" * 8
HANDLER_MARKER: Final[str] = "_memorysum_handler"
LOGGER_PREFIX: Final[str] = "memorysum_vl."

_SENSITIVE_KEY: Final[re.Pattern[str]] = re.compile(
    r"(token|secret|password|passwd|pwd|authorization|cookie|api[-_]?key|credential)",
    re.IGNORECASE,
)
_DATA_URL: Final[re.Pattern[str]] = re.compile(
    r"data:(?P<mime>[-\w.+]+/[-.\w.+]+);base64,(?P<payload>[A-Za-z0-9+/=\s]+)",
    re.IGNORECASE,
)
_BEARER_TOKEN: Final[re.Pattern[str]] = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}")
_CONTROL_CHAR: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")

# logging 自己维护的属性，不作为业务字段重复输出
_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "request_id",
        # uvicorn 附带的 ANSI 彩色模板，对人没有信息量
        "color_message",
    }
)

_request_id: ContextVar[str] = ContextVar("memorysum_request_id", default=NO_REQUEST_ID)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_request_id(value: str) -> Token[str]:
    return _request_id.set(value)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


def current_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """把当前请求的关联 ID 注入每条记录，让 service 层日志也能串起来。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = _request_id.get()
        return True


# --------------------------------------------------------------------------- 脱敏


def sanitize_text(value: str, *, limit: int | None = None) -> str:
    """抹掉 Bearer 凭据与 base64 图片体，折叠控制字符，必要时截断。"""
    if not value:
        return ""
    text = _DATA_URL.sub(_collapse_data_url, value)
    text = _BEARER_TOKEN.sub(f"Bearer {REDACTED}", text)
    text = _CONTROL_CHAR.sub(_escape_control, text)
    if limit is not None and len(text) > limit:
        omitted = len(text) - limit
        text = f"{text[:limit]}…<+{omitted} 字符>"
    return text


def sanitize_value(value: Any, *, limit: int = 512, _depth: int = 0) -> Any:
    """递归脱敏：敏感键整体替换，字符串走 `sanitize_text`。"""
    if _depth > 6:
        return "<层级过深>"
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _SENSITIVE_KEY.search(text_key):
                cleaned[text_key] = REDACTED
            else:
                cleaned[text_key] = sanitize_value(item, limit=limit, _depth=_depth + 1)
        return cleaned
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_value(item, limit=limit, _depth=_depth + 1) for item in value]
    if isinstance(value, str):
        return sanitize_text(value, limit=limit)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return sanitize_text(str(value), limit=limit)


def _collapse_data_url(match: re.Match[str]) -> str:
    payload = match.group("payload")
    return f"data:{match.group('mime')};base64,<已省略 {len(payload)} 字符>"


def _escape_control(match: re.Match[str]) -> str:
    char = match.group()
    if char == "\n":
        return "\\n"
    if char == "\r":
        return "\\r"
    if char == "\t":
        return "\\t"
    return f"\\x{ord(char):02x}"


# ------------------------------------------------------------------------- 格式化


class HumanFormatter(logging.Formatter):
    """`时间 级别 [请求ID] 模块 │ 正文`，续行沿用同一前缀。"""

    def format(self, record: logging.LogRecord) -> str:
        prefix = _prefix(record)
        message = record.getMessage()
        lines = [prefix + line for line in (message.splitlines() or [""])]
        extras = _extras(record)
        if extras:
            lines[0] = f"{lines[0]} · {' · '.join(extras)}"
        if record.exc_info is not None:
            traceback = self.formatException(record.exc_info)
            lines.extend(f"    {line}" for line in traceback.splitlines())
        return "\n".join(lines)


class JsonFormatter(logging.Formatter):
    """给下游采集用的单行 JSON，字段与人类可读格式一一对应。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _timestamp(record.created),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", NO_REQUEST_ID),
            "logger": _short_logger(record.name),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = sanitize_value(value, limit=2048)
        if record.exc_info is not None:
            payload["traceback"] = sanitize_text(
                self.formatException(record.exc_info), limit=16 * 1024
            )
        return json.dumps(payload, ensure_ascii=False, default=str)


def _prefix(record: logging.LogRecord) -> str:
    request_id = getattr(record, "request_id", NO_REQUEST_ID) or NO_REQUEST_ID
    return (
        f"{_timestamp(record.created)} {record.levelname:<5} "
        f"[{request_id}] {_short_logger(record.name)} │ "
    )


def _short_logger(name: str) -> str:
    return name.removeprefix(LOGGER_PREFIX) or name


def _timestamp(created: float) -> str:
    return datetime.fromtimestamp(created).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _extras(record: logging.LogRecord) -> list[str]:
    rendered: list[str] = []
    for key, value in record.__dict__.items():
        if key in _RESERVED or key.startswith("_"):
            continue
        rendered.append(f"{key}={_render_scalar(value)}")
    return rendered


def _render_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(sanitize_value(value), ensure_ascii=False, default=str)
    text = sanitize_text(str(value))
    if text and not any(char in text for char in ' ",='):
        return text
    return json.dumps(text, ensure_ascii=False)


# --------------------------------------------------------------------------- 配置


def configure_logging(settings: Settings) -> None:
    """安装 stderr（journald）与可选文件 handler，重复调用安全。"""
    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)
    correlation = RequestIdFilter()

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter = _build_formatter(settings.log_format)
    handlers: list[logging.Handler] = []
    if settings.log_file is not None:
        # 配了文件就只写文件，不往 stderr 同步一份，避免 journald 里重复堆积
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                settings.log_file,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding="utf-8",
            )
        )
    else:
        handlers.append(logging.StreamHandler(sys.stderr))

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(correlation)
        setattr(handler, HANDLER_MARKER, True)
        root.addHandler(handler)

    root.setLevel(level)

    # uvicorn 自带 handler 会把同一批记录再输出一遍，这里统一切到 root
    for name in ("uvicorn", "uvicorn.error"):
        target = logging.getLogger(name)
        for handler in list(target.handlers):
            target.removeHandler(handler)
            handler.close()
        target.propagate = True

    # access log 已由 RequestLoggingMiddleware 完整覆盖，避免同一请求打两遍
    access = logging.getLogger("uvicorn.access")
    for handler in list(access.handlers):
        access.removeHandler(handler)
        handler.close()
    access.propagate = False
    access.disabled = True

    # 第三方库的 INFO 噪音会淹没业务上下文
    for name in ("httpx", "httpcore", "urllib3", "asyncio", "multipart"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _build_formatter(style: str) -> logging.Formatter:
    return JsonFormatter() if style == "json" else HumanFormatter()
