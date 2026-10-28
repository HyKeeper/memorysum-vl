from memorysum_vl.domain.models import Conversation, WriteReceipt
from memorysum_vl.domain.ports import MemoryStore


class AddService:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def add(self, conversation: Conversation) -> WriteReceipt:
        await self._store.write(conversation)
        return WriteReceipt(
            request_id=conversation.request_id,
            user_id=conversation.user_id,
            session_id=conversation.session_id,
        )
