from collections.abc import Sequence

from memorysum_vl.core.logging import get_logger
from memorysum_vl.domain.models import MemoryEntry, SearchQuery
from memorysum_vl.domain.ports import MemoryStore

logger = get_logger(__name__)


class SearchService:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def search(self, query: SearchQuery) -> Sequence[MemoryEntry]:
        logger.info(
            "开始检索记忆",
            extra={
                "user_id": query.user_id,
                "top_k": query.top_k,
                "parts": len(query.parts),
                "options": ", ".join(query.options) or "none",
            },
        )
        entries = await self._store.retrieve(query)
        logger.info("检索完成", extra={"hits": len(entries), "user_id": query.user_id})
        return entries
