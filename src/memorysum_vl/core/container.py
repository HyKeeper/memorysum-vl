from dataclasses import dataclass

from memorysum_vl.adapters.stores.factory import build_memory_store
from memorysum_vl.core.auth import ApiToken, load_api_token
from memorysum_vl.core.config import Settings
from memorysum_vl.domain.ports import MemoryStore
from memorysum_vl.services.add_service import AddService
from memorysum_vl.services.search_service import SearchService


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    api_token: ApiToken
    memory_store: MemoryStore
    add_service: AddService
    search_service: SearchService


def build_container(settings: Settings) -> Container:
    memory_store = build_memory_store(settings)
    return Container(
        settings=settings,
        api_token=load_api_token(settings),
        memory_store=memory_store,
        add_service=AddService(memory_store),
        search_service=SearchService(memory_store),
    )
