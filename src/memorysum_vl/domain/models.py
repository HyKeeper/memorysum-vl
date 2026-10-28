from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str


@dataclass(frozen=True, slots=True)
class ImageContent:
    media_type: str
    data: bytes


Content = TextContent | ImageContent


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    parts: tuple[Content, ...]
    timestamp_ms: int | None = None


@dataclass(frozen=True, slots=True)
class Conversation:
    request_id: str
    user_id: str
    session_id: str
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class SearchQuery:
    parts: tuple[Content, ...]
    user_id: str
    top_k: int
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    id: str
    parts: tuple[Content, ...]
    score: float | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    request_id: str
    user_id: str
    session_id: str
