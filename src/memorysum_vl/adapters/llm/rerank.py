"""bge-reranker-v2-m3 客户端（llama.cpp `/rerank`，TEI 兼容格式）。

向量召回擅长「大概是这段」，交叉编码器擅长「就是这句」。所以重排只用在融合后
的头部候选上：多花几十毫秒，换 final 名次的精度，同时不碰长尾的延迟。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import httpx

from memorysum_vl.adapters.llm.transport import (
    RemoteModelError,
    post_json,
    probe_models,
)
from memorysum_vl.core.config import Settings
from memorysum_vl.core.logging import get_logger

logger = get_logger(__name__)

MAX_DOCUMENTS: Final[int] = 512


class BgeReranker:
    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.rerank_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.rerank_timeout, connect=3.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._model = settings.rerank_model
        self._retries = settings.rerank_retries

    async def aclose(self) -> None:
        await self._client.aclose()

    async def probe(self) -> str | None:
        return await probe_models(self._client, label="reranker")

    async def rerank(self, *, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        if len(documents) > MAX_DOCUMENTS:
            raise RemoteModelError(f"rerank supports at most {MAX_DOCUMENTS} documents")
        payload = {
            "model": self._model,
            "query": query,
            "documents": list(documents),
        }
        data = await post_json(
            self._client,
            "/rerank",
            payload,
            label="reranker",
            retries=self._retries,
        )
        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise RemoteModelError("reranker returned an unexpected payload")
        scores = [0.0] * len(documents)
        for item in results:
            try:
                index = int(item["index"])
                scores[index] = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RemoteModelError("reranker returned a malformed result") from exc
        return scores
