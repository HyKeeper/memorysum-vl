"""Rax-4.5 图片描述客户端。

图片在记忆里既要是「能看的证据」，也得是「能搜的文字」，所以描述的目标不是
写得漂亮，而是写成以后能被问题命中：主体、场景、动作、显眼物体、图内文字都要
落下来，且不许编。模型是带思维链的，这里用 `enable_thinking=false` 直接出正文，
既快（约 0.7 s/图）又不会被推理段落污染检索文本。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Final

import httpx

from memorysum_vl.adapters.llm.transport import (
    RemoteModelError,
    post_json,
    probe_models,
)
from memorysum_vl.core.config import Settings
from memorysum_vl.core.imaging import PreparedImage, to_data_uri
from memorysum_vl.core.logging import get_logger
from memorysum_vl.domain.models import Caption

logger = get_logger(__name__)

DEFAULT_PROMPT: Final[str] = (
    "请分两部分回答这张图：\n"
    "DESCRIPTION: 用 2-4 句客观描述画面内容（主体、场景、动作、显眼物体）。"
    "**必须单独说明整体背景与主色调**（例如「纯白背景」「深蓝渐变背景」「浅灰聊天界面」）。"
    "如果是截图、幻灯片、图表或文档，说明它是什么。不要臆造看不见的内容。\n"
    "TRANSCRIPT: 把图中所有可读文字按从上到下的顺序逐条抄写；聊天截图按「说话人：内容」"
    "一行一条，保留原文语言与顺序，日期、人名、数字要照抄。看不清的写 [unclear]。"
    "不要总结，不要遗漏。"
)

MAX_CAPTION_CHARS: Final[int] = 1200
MAX_TRANSCRIPT_CHARS: Final[int] = 4000
_TRANSCRIPT_LABEL: Final[re.Pattern[str]] = re.compile(r"(?i)^\s*transcript\s*[:：]\s*")

_THINK_BLOCK: Final[re.Pattern[str]] = re.compile(
    r"(?:◁think▷|◁think▷|<\s*think\s*>|<\|channel\|>\s*thought).*?"
    r"(?:◁/think▷|◁/think▷|<\s*/\s*think\s*>|<\|channel\|>)",
    re.DOTALL | re.IGNORECASE,
)
_CHANNEL_BLOCK: Final[re.Pattern[str]] = re.compile(r"<\|channel\|>.*?<\|channel\|>", re.DOTALL)
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


DESCRIPTION_MARK: Final[re.Pattern[str]] = re.compile(r"(?i)^\s*description\s*[:：]")
TRANSCRIPT_MARK: Final[re.Pattern[str]] = re.compile(r"(?i)transcript\s*[:：]")


class CaptionUnavailableError(RemoteModelError):
    pass


class RaxCaptioner:
    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.caption_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.caption_timeout, connect=3.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        self._model = settings.caption_model
        self._max_tokens = settings.caption_max_tokens
        self._retries = settings.caption_retries
        self._prompt = DEFAULT_PROMPT
        self._semaphore = asyncio.Semaphore(max(1, settings.caption_concurrency))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def probe(self) -> str | None:
        return await probe_models(self._client, label="captioner")

    async def caption(self, images: Sequence[PreparedImage]) -> list[Caption]:
        if not images:
            return []
        return list(await asyncio.gather(*(self._caption_one(image) for image in images)))

    async def _caption_one(self, image: PreparedImage) -> Caption:
        attempts = (self._max_tokens, self._max_tokens * 2)
        last_error: Exception | None = None
        for max_tokens in attempts:
            try:
                return await self._request(image, max_tokens=max_tokens)
            except CaptionUnavailableError as exc:
                last_error = exc
                logger.debug("图片描述为空，放宽长度重试", extra={"error": str(exc)})
        raise CaptionUnavailableError(str(last_error) if last_error else "captioner failed")

    async def _request(self, image: PreparedImage, *, max_tokens: int) -> Caption:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": to_data_uri(image.media_type, image.data),
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with self._semaphore:
            data = await post_json(
                self._client,
                "/v1/chat/completions",
                payload,
                label="captioner",
                retries=self._retries,
            )
        message = (data.get("choices") or [{}])[0].get("message") or {}
        text = _THINK_BLOCK.sub(" ", str(message.get("content") or ""))
        parsed = parse_caption(text)
        if not parsed.description and not parsed.transcript:
            raise CaptionUnavailableError("captioner returned empty content")
        return parsed


def parse_caption(raw: str) -> Caption:
    """拆 `DESCRIPTION:` / `TRANSCRIPT:` 两段；没有标记就整段当描述。"""
    text = _CHANNEL_BLOCK.sub(" ", raw).strip()
    match = TRANSCRIPT_MARK.search(text)
    if match is None:
        description = DESCRIPTION_MARK.sub("", text, count=1)
        return Caption(description=clean_text(description), transcript="")
    description = DESCRIPTION_MARK.sub("", text[: match.start()], count=1)
    transcript = text[match.end() :]
    transcript = _TRANSCRIPT_LABEL.sub("", transcript, count=1)
    return Caption(description=clean_text(description), transcript=clean_transcript(transcript))


def clean_text(raw: str) -> str:
    text = _WHITESPACE.sub(" ", raw).strip(" -—:*\t")
    if len(text) > MAX_CAPTION_CHARS:
        text = text[:MAX_CAPTION_CHARS].rstrip()
    return text


def clean_transcript(raw: str) -> str:
    lines = [_WHITESPACE.sub(" ", line).strip(" -*\t") for line in raw.splitlines()]
    kept = [line for line in lines if line and line != "[unclear]"]
    text = " ".join(kept).strip()
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS].rstrip()
    return text
