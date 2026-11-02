"""测试替身与装配。

约定：测试只保留少量端到端用例（见 `test_e2e.py`），不为每个函数写单测。
所以这里的替身要尽量「像真的」：假嵌入是稳定哈希的词袋向量，词面重叠就能被
余弦相似度体现；图片向量取该图描述的文本向量，于是文本问句能命中图片记忆，
图片查询也能命中自己的图片。
"""

from __future__ import annotations

import hashlib
import math
import zlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import httpx
import pytest

from memorysum_vl.adapters.llm.transport import RemoteModelError
from memorysum_vl.adapters.stores.repository import SqliteRepository
from memorysum_vl.adapters.stores.sqlite_store import SqliteMemoryStore
from memorysum_vl.core.auth import ApiToken
from memorysum_vl.core.config import Settings
from memorysum_vl.core.container import Container
from memorysum_vl.core.imaging import PreparedImage
from memorysum_vl.core.text import cjk_bigrams, query_terms, tokens
from memorysum_vl.domain.models import Caption
from memorysum_vl.domain.ports import Vector
from memorysum_vl.main import create_app
from memorysum_vl.services.add_service import AddService
from memorysum_vl.services.search_service import SearchService

DIM = 64
TOKEN = "test-token"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def features(text: str) -> list[str]:
    """拉丁词做轻量词干化，中文补二元组——让假模型在两种语言下都「有点语义」。"""
    result: list[str] = []
    for token in tokens(text.lower()):
        if token.isascii():
            stem = token
            for suffix in ("ing", "ed", "es", "s"):
                if len(token) > len(suffix) + 3 and token.endswith(suffix):
                    stem = token[: -len(suffix)]
                    break
            result.append(stem)
        else:
            result.append(token)
            result.extend(cjk_bigrams(token))
    return result


class FakeVision:
    """描述注册表：captioner 先写、embedder 再读，让图片有了「文字指纹」。"""

    def __init__(self) -> None:
        self.captions: dict[str, str] = {}
        self.transcripts: dict[str, str] = {}
        self.caption_calls = 0


class FakeEmbedder:
    def __init__(self, vision: FakeVision, *, dim: int = DIM) -> None:
        self.vision = vision
        self.dim = dim
        self.fail = False

    def _vector(self, text: str) -> Vector:
        buckets = [0.0] * self.dim
        for token in features(text):
            code = zlib.crc32(token.encode("utf-8"))
            buckets[code % self.dim] += 1.0 if (code >> 8) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
        return tuple(value / norm for value in buckets)

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        if self.fail:
            raise RemoteModelError("embedding service down")
        return [self._vector(text) for text in texts]

    async def embed_images(self, images: Sequence[PreparedImage]) -> list[Vector]:
        if self.fail:
            raise RemoteModelError("embedding service down")
        return [self._vector(self._image_text(image)) for image in images]

    async def embed_multimodal(self, *, text: str, images: Sequence[PreparedImage]) -> Vector:
        if self.fail:
            raise RemoteModelError("embedding service down")
        merged = " ".join([text, *(self._image_text(image) for image in images)])
        return self._vector(merged)

    def _image_text(self, image: PreparedImage) -> str:
        return self.vision.captions.get(_digest(image.data), "unlabeled image")


class FakeCaptioner:
    def __init__(
        self, vision: FakeVision, *, default: str = "a red bicycle parked in the park"
    ) -> None:
        self.vision = vision
        self.default_caption = default
        self.fail = False

    async def caption(self, images: Sequence[PreparedImage]) -> list[Caption]:
        if self.fail:
            raise RemoteModelError("captioner down")
        captions: list[Caption] = []
        for image in images:
            self.vision.caption_calls += 1
            digest = _digest(image.data)
            description = self.vision.captions.setdefault(digest, self.default_caption)
            transcript = self.vision.transcripts.get(digest, "")
            captions.append(Caption(description=description, transcript=transcript))
        return captions


def scoring_terms(text: str) -> set[str]:
    """重排只看实词：停用词不参与，避免「is/the」把分数抬起来。"""
    return {token for term in query_terms(text) for token in features(term)}


class FakeReranker:
    def __init__(self) -> None:
        self.fail = False

    async def rerank(self, *, query: str, documents: Sequence[str]) -> list[float]:
        if self.fail:
            raise RemoteModelError("reranker down")
        wanted = scoring_terms(query)
        return [
            len(wanted & scoring_terms(document)) / max(len(wanted), 1) for document in documents
        ]


@dataclass
class Harness:
    store: SqliteMemoryStore
    repository: SqliteRepository
    embedder: FakeEmbedder
    captioner: FakeCaptioner
    reranker: FakeReranker
    vision: FakeVision


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "memory.db",
        embedding_dim=DIM,
        route_pool=50,
        rerank_top_n=20,
        log_file=None,
        auth_token=TOKEN,
        caption_concurrency=2,
        embedding_concurrency=2,
        image_max_side=256,
    )


@pytest.fixture
def vision() -> FakeVision:
    return FakeVision()


@pytest.fixture
async def harness(settings: Settings, vision: FakeVision) -> AsyncIterator[Harness]:
    repository = SqliteRepository(settings.db_path, embedding_dim=settings.embedding_dim)
    embedder = FakeEmbedder(vision)
    captioner = FakeCaptioner(vision)
    reranker = FakeReranker()
    store = SqliteMemoryStore(
        settings=settings,
        repository=repository,
        embedder=embedder,
        captioner=captioner,
        reranker=reranker,
    )
    bundle = Harness(
        store=store,
        repository=repository,
        embedder=embedder,
        captioner=captioner,
        reranker=reranker,
        vision=vision,
    )
    yield bundle
    await store.aclose()


@pytest.fixture
def app(settings: Settings, harness: Harness):
    container = Container(
        settings=settings,
        api_token=ApiToken(secret=TOKEN),
        memory_store=harness.store,
        add_service=AddService(harness.store),
        search_service=SearchService(harness.store),
    )
    return create_app(settings, container)


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://memory.test",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as instance:
        yield instance
