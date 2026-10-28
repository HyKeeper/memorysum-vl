from collections.abc import Sequence

from memorysum_vl.domain.models import MemoryEntry, SearchQuery
from memorysum_vl.domain.ports import MemoryStore


class SearchService:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def search(self, query: SearchQuery) -> Sequence[MemoryEntry]:
        return await self._store.retrieve(query)
