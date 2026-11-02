"""Qwen3-VL-Embedding-2B 客户端。

llama.cpp 的 `/v1/embeddings` 除了字符串数组，还认一种非 OpenAI 的条目：

    {"prompt_string": "<__media__>\\n文本", "multimodal_data": ["<图片原始字节的 base64>"]}

`<__media__>` 是 server 的多模态占位符，默认每次启动随机生成，所以启动脚本里
用 `LLAMA_MEDIA_MARKER` 把它钉住，这里再按同一个值拼 prompt。

文本与图片落在同一个空间，因此同一批向量既能做「文字问 → 文字记忆」，也能做
「文字问 → 图片记忆」和「图片问 → 文字记忆」，不需要额外对齐层。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

import httpx

from memorysum_vl.adapters.llm.transport import (
    RemoteModelError,
    post_json,
    probe_models,
)
from memorysum_vl.core.config import Settings
from memorysum_vl.core.imaging import PreparedImage, to_base64
from memorysum_vl.core.logging import get_logger
from memorysum_vl.domain.ports import Vector

logger = get_logger(__name__)

CHATML_TEMPLATE: Final[str] = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{content}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
SPECIAL_TOKEN_PREFIX: Final[str] = "<|"


class EmbeddingDimensionError(RuntimeError):
    pass


class QwenVLEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.embedding_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.embedding_timeout, connect=3.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim
        self._batch = max(1, settings.embedding_batch_size)
        self._retries = settings.embedding_retries
        self._instruction = settings.embedding_instruction.strip()
        self._chat_template = settings.embedding_chat_template
        self._marker = settings.embedding_media_marker
        self._semaphore = asyncio.Semaphore(max(1, settings.embedding_concurrency))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def probe(self) -> str | None:
        return await probe_models(self._client, label="embedding")

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        batches = [
            list(texts[start : start + self._batch]) for start in range(0, len(texts), self._batch)
        ]
        results = await asyncio.gather(*(self._embed_text_batch(batch) for batch in batches))
        vectors = [vector for batch in results for vector in batch]
        if len(vectors) != len(texts):
            raise RemoteModelError(
                f"embedding server returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    async def embed_images(self, images: Sequence[PreparedImage]) -> list[Vector]:
        if not images:
            return []
        batches = [
            list(images[start : start + self._batch])
            for start in range(0, len(images), self._batch)
        ]
        results = await asyncio.gather(*(self._embed_image_batch(batch) for batch in batches))
        vectors = [vector for batch in results for vector in batch]
        if len(vectors) != len(images):
            raise RemoteModelError(
                f"embedding server returned {len(vectors)} vectors for {len(images)} images"
            )
        return vectors

    async def embed_multimodal(self, *, text: str, images: Sequence[PreparedImage]) -> Vector:
        if not images:
            return (await self.embed_texts([text]))[0]
        media = "".join(f"{self._marker}\n" for _ in images)
        prompt = self._compose(text, media=media)
        payload = {
            "input": [
                {
                    "prompt_string": prompt,
                    "multimodal_data": [to_base64(image) for image in images],
                }
            ],
            "model": self._model,
        }
        async with self._semaphore:
            data = await post_json(
                self._client,
                "/v1/embeddings",
                payload,
                label="embedding.multimodal",
                retries=self._retries,
            )
        return self._vector(data, expected=1)[0]

    async def _embed_text_batch(self, texts: list[str]) -> list[Vector]:
        payload = {"input": [self._compose(text) for text in texts], "model": self._model}
        async with self._semaphore:
            data = await post_json(
                self._client,
                "/v1/embeddings",
                payload,
                label="embedding.text",
                retries=self._retries,
            )
        return self._vector(data, expected=len(texts))

    async def _embed_image_batch(self, images: list[PreparedImage]) -> list[Vector]:
        payload = {
            "input": [
                {
                    "prompt_string": self._compose("", media=f"{self._marker}\n"),
                    "multimodal_data": [to_base64(image)],
                }
                for image in images
            ],
            "model": self._model,
        }
        async with self._semaphore:
            data = await post_json(
                self._client,
                "/v1/embeddings",
                payload,
                label="embedding.image",
                retries=self._retries,
            )
        return self._vector(data, expected=len(images))

    def _compose(self, content: str, *, media: str = "") -> str:
        """拼模型输入；`media` 里的占位符由调用方给，不能经过 neutralize。"""
        cleaned = f"{media}{neutralize(content)}"
        if not self._chat_template or not self._instruction:
            return cleaned
        return CHATML_TEMPLATE.format(instruction=self._instruction, content=cleaned)

    def _vector(self, data: dict, *, expected: int) -> list[Vector]:
        items = data.get("data")
        if not isinstance(items, list) or len(items) != expected:
            raise RemoteModelError(
                f"embedding server returned an unexpected payload for {expected} inputs"
            )
        vectors: list[Vector] = []
        for item in sorted(items, key=lambda entry: int(entry.get("index", 0))):
            raw = item.get("embedding")
            if not isinstance(raw, list) or not raw:
                raise RemoteModelError("embedding server returned an empty embedding")
            if len(raw) != self._dim:
                raise EmbeddingDimensionError(
                    f"embedding dim {len(raw)} != MEMORYSUM_EMBEDDING_DIM {self._dim}; "
                    "restart the server with the model you configured (or fix the setting)"
                )
            vectors.append(tuple(float(value) for value in raw))
        return vectors


def neutralize(text: str) -> str:
    """把可能被当成控制符的串降级成普通字符，避免用户文本改写 ChatML 结构。"""
    return text.replace(SPECIAL_TOKEN_PREFIX, "< |").replace("<__media__>", "[media]")
