"""Reciprocal Rank Fusion：只吃排名，不吃分数的量纲。

多路召回的分数天然不可比——余弦相似度在 0..1，BM25 是无界负值，跨编码器
logit 可以到 ±20。直接加权求和等于把权重交给方差最大的那一路。RRF 把
「第 rank 名」折算成 `w / (k + rank)`，k 越大越平滑：k=20 时第 1 名贡献 1/21，
第 20 名贡献 1/40，既保留头部优势，又不至于让某一路的单点噪声决定排序。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

DEFAULT_K: Final[int] = 20


@dataclass(frozen=True, slots=True)
class RouteRanking:
    """一路召回的排名结果，按相关性从高到低。"""

    name: str
    ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FusedHit:
    id: str
    score: float
    best_rank: int
    routes: tuple[str, ...]


@dataclass(slots=True)
class _Accumulator:
    score: float = 0.0
    best_rank: int = 0
    routes: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    rankings: Sequence[RouteRanking],
    *,
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """融合多路排名，同 id 只计一次，重复出现只会多拿一份贡献。

    权重按路名配置，缺省为 1.0；权重 ≤0 的路直接跳过（离线调参时更方便）。
    """
    if k < 1:
        raise ValueError(f"rrf k must be >= 1, got {k}")

    acc: dict[str, _Accumulator] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        weight = 1.0 if weights is None else float(weights.get(ranking.name, 1.0))
        if weight <= 0:
            continue
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking.ids):
            if item_id in seen:
                continue
            seen.add(item_id)
            slot = acc.get(item_id)
            if slot is None:
                slot = _Accumulator(best_rank=rank)
                acc[item_id] = slot
                order[item_id] = len(order)
            slot.score += weight / (k + rank + 1)
            slot.best_rank = min(slot.best_rank, rank)
            slot.routes.append(ranking.name)

    fused = [
        FusedHit(
            id=item_id,
            score=slot.score,
            best_rank=slot.best_rank,
            routes=tuple(slot.routes),
        )
        for item_id, slot in acc.items()
    ]
    fused.sort(key=lambda hit: (-hit.score, hit.best_rank, order[hit.id]))
    return fused if limit is None else fused[:limit]


def sigmoid(value: float) -> float:
    """把跨编码器 logit 压到 (0, 1)，保序，便于当作可读的 score 返回。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exp = math.exp(max(value, -60.0))
    return exp / (1.0 + exp)


def normalize_scores(values: Sequence[float]) -> list[float]:
    """按最大值归一，用于没有重排时的兜底分数。"""
    if not values:
        return []
    top = max(values)
    if top <= 0:
        return [0.0 for _ in values]
    return [value / top for value in values]


def adaptive_k(
    candidate_count: int, *, base: int = DEFAULT_K, low: int = 10, high: int = 30
) -> int:
    """候选少时收紧 k（头部优势更大），候选多时放宽。

    只有 6 条候选时 k=20 会把第 1 名和第 6 名压得几乎一样（1/21 vs 1/26），
    排序失去区分度；按候选量的三分之一取 k 能让小候选集和大队列都合理。
    """
    if candidate_count <= 0:
        return base
    return max(low, min(high, candidate_count // 3 or base))


def fuse_rerank_ranks(
    fused: Sequence[FusedHit],
    rerank_scores: Sequence[float],
    *,
    k: int,
    fusion_weight: float,
) -> list[tuple[int, float]]:
    """把精排名次与召回名次再融合一次。

    `score = 1/(k+rerank_rank) + w/(k+retrieval_rank)`：cross-encoder 只决定
    邻近名次的先后，多路召回的共识仍然算数。纯按 rerank 分数排序时，一次
    噪声打分就能把正确的证据压到很后面（图搜图那次就是这样翻车的）。
    """
    if len(fused) != len(rerank_scores):
        raise ValueError("fused hits and rerank scores must align")
    retrieval_rank = {index: rank + 1 for rank, index in enumerate(range(len(fused)))}
    order = sorted(range(len(fused)), key=lambda index: (-rerank_scores[index], index))
    rerank_rank = {index: rank + 1 for rank, index in enumerate(order)}
    scored = [
        (
            index,
            1.0 / (k + rerank_rank[index]) + fusion_weight / (k + retrieval_rank[index]),
        )
        for index in range(len(fused))
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored
