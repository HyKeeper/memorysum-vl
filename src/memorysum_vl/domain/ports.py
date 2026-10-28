from collections.abc import Sequence
from typing import Protocol

from memorysum_vl.domain.models import Conversation, MemoryEntry, SearchQuery


class MemoryWriter(Protocol):
    async def write(self, conversation: Conversation) -> None: ...


class MemoryRetriever(Protocol):
    async def retrieve(self, query: SearchQuery) -> Sequence[MemoryEntry]: ...


class MemoryStore(MemoryWriter, MemoryRetriever, Protocol):
    pass
