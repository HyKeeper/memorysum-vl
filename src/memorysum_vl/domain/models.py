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


@dataclass(frozen=True, slots=True)
class Caption:
    """一张图的两种文字证据：客观描述（入向量）与逐条转写（入词法）。"""

    description: str
    transcript: str = ""


@dataclass(frozen=True, slots=True)
class MemoryImage:
    """入库后的一张图：原始字节 + 视觉模型给的描述。"""

    sha256: str
    media_type: str
    data: bytes
    caption: str = ""


@dataclass(frozen=True, slots=True)
class MemoryUnit:
    """最小记忆单元 = 一条消息。

    粒度选在「一条消息」而不是「一次 Add」：一次 Add 可能带几十条消息，
    整块检索会把答案稀释成长文档，而评测里的问题通常只对应其中一两句。
    """

    id: str
    user_id: str
    session_id: str
    request_id: str
    ordinal: int
    role: Role
    timestamp_ms: int | None
    created_at: datetime
    text: str
    captions: tuple[str, ...] = ()
    transcripts: tuple[str, ...] = ()
    images: tuple[MemoryImage, ...] = ()
    caption_state: str = "ok"
    embedding_state: str = "ok"
