from collections.abc import Sequence

from memorysum_vl.domain.models import Conversation, MemoryEntry, SearchQuery


class InMemoryMemoryStore:
    async def write(self, conversation: Conversation) -> None:
        raise NotImplementedError("conversation persistence is not implemented")

    async def retrieve(self, query: SearchQuery) -> Sequence[MemoryEntry]:
        raise NotImplementedError("memory retrieval is not implemented")
