from dataclasses import dataclass

from memorysum_vl.adapters.stores.factory import StoreBundle, build_memory_store
from memorysum_vl.core.auth import ApiToken, load_api_token
from memorysum_vl.core.config import Settings
from memorysum_vl.core.logging import get_logger
from memorysum_vl.domain.ports import MemoryStore
from memorysum_vl.services.add_service import AddService
from memorysum_vl.services.search_service import SearchService

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    api_token: ApiToken
    memory_store: MemoryStore
    add_service: AddService
    search_service: SearchService
    bundle: StoreBundle | None = None

    async def aclose(self) -> None:
        closer = getattr(self.memory_store, "aclose", None)
        if closer is not None:
            await closer()
        if self.bundle is None:
            return
        await self.bundle.embedder.aclose()
        await self.bundle.captioner.aclose()
        await self.bundle.reranker.aclose()

    async def probe(self) -> dict[str, str | None]:
        """启动时探一次三个模型服务，只写日志，不阻塞请求。"""
        if self.bundle is None:
            return {}
        embedding = await self.bundle.embedder.probe()
        captioning = await self.bundle.captioner.probe()
        reranking = await self.bundle.reranker.probe()
        return {"embedding": embedding, "captioner": captioning, "reranker": reranking}


def build_container(settings: Settings) -> Container:
    bundle = build_memory_store(settings)
    return Container(
        settings=settings,
        api_token=load_api_token(settings),
        memory_store=bundle.store,
        add_service=AddService(bundle.store),
        search_service=SearchService(bundle.store),
        bundle=bundle,
    )
