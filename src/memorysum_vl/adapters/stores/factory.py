"""装配真实存储：SQLite + 三个本机模型服务。

所有可替换点（Embedder / Captioner / Reranker）都只通过 `domain.ports` 里的协议
出现，换模型只需换这里的一行，不用碰检索流程。
"""

from __future__ import annotations

from dataclasses import dataclass

from memorysum_vl.adapters.llm.caption import RaxCaptioner
from memorysum_vl.adapters.llm.embedding import QwenVLEmbedder
from memorysum_vl.adapters.llm.rerank import BgeReranker
from memorysum_vl.adapters.stores.repository import SqliteRepository
from memorysum_vl.adapters.stores.sqlite_store import SqliteMemoryStore
from memorysum_vl.core.config import Settings
from memorysum_vl.domain.ports import MemoryStore


@dataclass(frozen=True, slots=True)
class StoreBundle:
    store: MemoryStore
    embedder: QwenVLEmbedder
    captioner: RaxCaptioner
    reranker: BgeReranker


def build_memory_store(settings: Settings) -> StoreBundle:
    repository = SqliteRepository(settings.db_path, embedding_dim=settings.embedding_dim)
    embedder = QwenVLEmbedder(settings)
    captioner = RaxCaptioner(settings)
    reranker = BgeReranker(settings)
    store = SqliteMemoryStore(
        settings=settings,
        repository=repository,
        embedder=embedder,
        captioner=captioner,
        reranker=reranker,
    )
    return StoreBundle(store=store, embedder=embedder, captioner=captioner, reranker=reranker)
