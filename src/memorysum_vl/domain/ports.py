from collections.abc import Sequence
from typing import Protocol

from memorysum_vl.core.imaging import PreparedImage
from memorysum_vl.domain.models import Caption, Conversation, MemoryEntry, SearchQuery

type Vector = tuple[float, ...]


class MemoryWriter(Protocol):
    async def write(self, conversation: Conversation) -> None: ...


class MemoryRetriever(Protocol):
    async def retrieve(self, query: SearchQuery) -> Sequence[MemoryEntry]: ...


class MemoryStore(MemoryWriter, MemoryRetriever, Protocol):
    pass


class Embedder(Protocol):
    """文本与图片共用一个向量空间，模态之间可以直接比。"""

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]: ...

    async def embed_images(self, images: Sequence[PreparedImage]) -> list[Vector]: ...

    async def embed_multimodal(self, *, text: str, images: Sequence[PreparedImage]) -> Vector: ...


class Captioner(Protocol):
    """把图片转成客观描述，作为可检索的文字证据。"""

    async def caption(self, images: Sequence[PreparedImage]) -> list[Caption]: ...


class Reranker(Protocol):
    """交叉编码器重排：query 与候选逐对打分，保序返回分数。"""

    async def rerank(self, *, query: str, documents: Sequence[str]) -> list[float]: ...


class AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


__all__ = [
    "AsyncClosable",
    "Captioner",
    "Embedder",
    "MemoryRetriever",
    "MemoryStore",
    "MemoryWriter",
    "Reranker",
    "Vector",
]
