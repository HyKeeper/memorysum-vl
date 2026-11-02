"""模型服务的 HTTP 通道：统一的超时、重试与错误语义。

三路模型都在本机 llama.cpp 上，正常调用是毫秒级；这里只兜住「重启中 / 偶发
5xx / 连接被重置」这类瞬时故障。重试预算刻意压得很小，避免把评测的 Add 拖长。
"""

from __future__ import annotations

import asyncio
from typing import Final

import httpx

from memorysum_vl.core.logging import get_logger

logger = get_logger(__name__)

RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504, 524})
RETRY_DELAY: Final[float] = 0.2


class RemoteModelError(RuntimeError):
    """上游模型调用失败（已用尽重试）。"""


async def post_json(
    client: httpx.AsyncClient,
    path: str,
    payload: dict,
    *,
    label: str,
    retries: int = 1,
) -> dict:
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.post(path, json=payload)
            if response.status_code in RETRYABLE_STATUS:
                last_error = RemoteModelError(
                    f"{label} returned {response.status_code}: {response.text[:200]}"
                )
            elif response.status_code >= 400:
                raise RemoteModelError(
                    f"{label} rejected the request ({response.status_code}): {response.text[:300]}"
                )
            else:
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            logger.debug(
                "模型调用重试",
                extra={"label": label, "attempt": attempt + 1, "error": type(last_error).__name__},
            )
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    raise RemoteModelError(
        f"{label} failed after {attempts} attempts: {last_error}"
    ) from last_error


async def probe_models(client: httpx.AsyncClient, *, label: str) -> str | None:
    """探活并取回服务端模型名，启动日志里用。"""
    try:
        response = await client.get("/v1/models", timeout=3.0)
        if response.status_code >= 400:
            return None
        models = response.json().get("data") or []
        if models:
            return str(models[0].get("id") or label)
    except (httpx.HTTPError, ValueError):
        return None
    return None
