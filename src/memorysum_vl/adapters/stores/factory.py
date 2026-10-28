from memorysum_vl.adapters.stores.in_memory import InMemoryMemoryStore
from memorysum_vl.core.config import Settings
from memorysum_vl.domain.ports import MemoryStore


def build_memory_store(settings: Settings) -> MemoryStore:
    return InMemoryMemoryStore()
