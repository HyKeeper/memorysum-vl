from memorysum_vl.core.logging import get_logger
from memorysum_vl.domain.models import Conversation, WriteReceipt
from memorysum_vl.domain.ports import MemoryStore

logger = get_logger(__name__)


class AddService:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def add(self, conversation: Conversation) -> WriteReceipt:
        logger.info(
            "开始写入会话",
            extra={
                "client_request_id": conversation.request_id,
                "user_id": conversation.user_id,
                "session_id": conversation.session_id,
                "messages": len(conversation.messages),
            },
        )
        await self._store.write(conversation)
        logger.info("会话写入完成", extra={"client_request_id": conversation.request_id})
        return WriteReceipt(
            request_id=conversation.request_id,
            user_id=conversation.user_id,
            session_id=conversation.session_id,
        )
