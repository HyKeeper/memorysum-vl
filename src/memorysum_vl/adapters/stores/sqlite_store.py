"""多模态记忆存储：写入时补描述与向量，检索时多路召回 → RRF → 重排。

一次 Add 内的工作流：

    消息 → 图片指纹/缩放 → 描述缓存命中检查 → 补描述 → 落库(原文+原图+FTS) → 写向量

一次 Search 内的工作流：

    查询 → 词面两路 + 文本/图片向量多路 → RRF(k=20) → bge 重排头部 → top_k

降级是显式的：描述服务挂了就存空描述并标 `caption_state=missing`，嵌入服务挂了
就标 `embedding_state=missing`，但消息本身照样能被 BM25 与重排搜到，Add 仍然返回
成功——比赛里「少一路」远好于「整条写入失败」。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from memorysum_vl.adapters.llm.transport import RemoteModelError
from memorysum_vl.adapters.stores.repository import (
    CachedCaption,
    CaptionRecord,
    SqliteRepository,
    StoredUnit,
    WriteResult,
)
from memorysum_vl.core.config import Settings
from memorysum_vl.core.fusion import (
    RouteRanking,
    adaptive_k,
    fuse_rerank_ranks,
    normalize_scores,
    reciprocal_rank_fusion,
    sigmoid,
)
from memorysum_vl.core.imaging import PreparedImage, prepare_for_model, sha256_hex
from memorysum_vl.core.logging import get_logger
from memorysum_vl.core.media import InlineImage
from memorysum_vl.core.render import render_text
from memorysum_vl.core.text import (
    clip_middle,
    salient_terms,
    strip_option_label,
    trigram_query,
    truncate,
    word_query,
    zh_index_text,
    zh_query,
)
from memorysum_vl.domain.models import (
    Content,
    Conversation,
    ImageContent,
    MemoryEntry,
    MemoryImage,
    MemoryUnit,
    Message,
    SearchQuery,
    TextContent,
)
from memorysum_vl.domain.ports import Captioner, Embedder, Reranker

logger = get_logger(__name__)

ID_PREFIX: Final[str] = "mem_"
ID_DIGEST_CHARS: Final[int] = 20


@dataclass(frozen=True, slots=True)
class _PreparedWrite:
    units: tuple[MemoryUnit, ...]
    lexicals: tuple[str, ...]
    lexicals_full: tuple[str, ...]
    lexicals_zh: tuple[str, ...]
    prepared_images: dict[str, PreparedImage]
    caption_count: int


class SqliteMemoryStore:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: SqliteRepository,
        embedder: Embedder,
        captioner: Captioner,
        reranker: Reranker,
    ) -> None:
        self._settings = settings
        self._repo = repository
        self._embedder = embedder
        self._captioner = captioner
        self._reranker = reranker
        self._image_cache: dict[str, PreparedImage] = {}

    async def aclose(self) -> None:
        await asyncio.to_thread(self._repo.close)

    # ---------------------------------------------------------------------- 写入

    async def write(self, conversation: Conversation) -> None:
        prepared = await self._prepare_write(conversation)
        result = await asyncio.to_thread(
            self._repo.replace_request,
            user_id=conversation.user_id,
            session_id=conversation.session_id,
            request_id=conversation.request_id,
            units=prepared.units,
            lexicals=prepared.lexicals,
            lexicals_full=prepared.lexicals_full,
            lexicals_zh=prepared.lexicals_zh,
        )
        logger.info(
            "记忆已落库",
            extra={
                "units": len(result.units),
                "images": len(result.images),
                "captions": prepared.caption_count,
            },
        )
        await self._index_vectors(conversation.user_id, prepared, result)

    async def _prepare_write(self, conversation: Conversation) -> _PreparedWrite:
        parts: list[ImageContent] = [
            part
            for message in conversation.messages
            for part in message.parts
            if isinstance(part, ImageContent)
        ]
        occurrences = await asyncio.to_thread(_prepare_occurrences, parts, self._settings)
        prepared_images: dict[str, PreparedImage] = {}
        shas: list[str] = []
        for digest, prepared in occurrences:
            prepared_images.setdefault(digest, prepared)
            shas.append(digest)

        captions = await self._resolve_captions(prepared_images)
        created_at = datetime.now(tz=UTC)

        units: list[MemoryUnit] = []
        lexicals: list[str] = []
        lexicals_full: list[str] = []
        lexicals_zh: list[str] = []
        index = 0
        for ordinal, message in enumerate(conversation.messages):
            text = _message_text(message)
            unit_captions: list[str] = []
            unit_transcripts: list[str] = []
            images: list[MemoryImage] = []
            for part in message.parts:
                if not isinstance(part, ImageContent):
                    continue
                digest = shas[index]
                index += 1
                cached = captions.get(digest)
                caption = cached.description if cached else ""
                transcript = cached.transcript if cached else ""
                unit_captions.append(caption)
                unit_transcripts.append(transcript)
                images.append(
                    MemoryImage(
                        sha256=digest,
                        media_type=part.media_type,
                        data=part.data,
                        caption=caption,
                    )
                )
            has_missing = any(image.caption == "" for image in images)
            unit = MemoryUnit(
                id=_unit_id(
                    conversation.user_id,
                    conversation.session_id,
                    conversation.request_id,
                    ordinal,
                ),
                user_id=conversation.user_id,
                session_id=conversation.session_id,
                request_id=conversation.request_id,
                ordinal=ordinal,
                role=message.role,
                timestamp_ms=message.timestamp_ms,
                created_at=created_at,
                text=text,
                captions=tuple(unit_captions),
                transcripts=tuple(unit_transcripts),
                images=tuple(images),
                caption_state="missing" if has_missing else "ok",
            )
            units.append(unit)
            lexical = render_text(
                role=unit.role,
                timestamp_ms=unit.timestamp_ms,
                text=unit.text,
                captions=unit.captions,
                include_transcripts=False,
            )
            lexical_full = render_text(
                role=unit.role,
                timestamp_ms=unit.timestamp_ms,
                text=unit.text,
                captions=unit.captions,
                transcripts=unit.transcripts,
            )
            lexicals.append(lexical)
            lexicals_full.append(lexical_full)
            lexicals_zh.append(zh_index_text(lexical_full))
        return _PreparedWrite(
            units=tuple(units),
            lexicals=tuple(lexicals),
            lexicals_full=tuple(lexicals_full),
            lexicals_zh=tuple(lexicals_zh),
            prepared_images=prepared_images,
            caption_count=len(captions),
        )

    async def _resolve_captions(
        self, prepared: dict[str, PreparedImage]
    ) -> dict[str, CachedCaption]:
        if not prepared:
            return {}
        cached = (
            await asyncio.to_thread(self._repo.lookup_captions, list(prepared))
            if self._settings.caption_cache
            else {}
        )
        missing = [
            digest for digest in prepared if not (cached.get(digest) and cached[digest].description)
        ]
        if not missing:
            return cached
        try:
            generated = await self._captioner.caption([prepared[digest] for digest in missing])
        except RemoteModelError as exc:
            logger.warning(
                "图片描述不可用，先按原文入库",
                extra={"images": len(missing), "error": type(exc).__name__},
            )
            return cached

        records: list[CaptionRecord] = []
        for digest, caption in zip(missing, generated, strict=True):
            description = caption.description.strip()
            if not description:
                continue
            cached[digest] = CachedCaption(
                description=description, transcript=caption.transcript.strip()
            )
            records.append(
                CaptionRecord(
                    sha256=digest,
                    media_type=prepared[digest].media_type,
                    caption=description,
                    transcript=caption.transcript.strip(),
                )
            )
        if records and self._settings.caption_cache:
            await asyncio.to_thread(
                self._repo.store_captions, records, model=self._settings.caption_model
            )
        return cached

    async def _index_vectors(
        self, user_id: str, prepared: _PreparedWrite, result: WriteResult
    ) -> None:
        text_vectors: list[Sequence[float]] | None = None
        try:
            text_vectors = await self._embedder.embed_texts(
                [stored.lexical for stored in result.units]
            )
        except RemoteModelError as exc:
            logger.warning(
                "文本向量不可用，仅保留词面索引",
                extra={"units": len(result.units), "error": type(exc).__name__},
            )
            await asyncio.to_thread(
                self._repo.mark_states,
                unit_ids=[stored.id for stored in result.units],
                column="embedding_state",
                state="missing",
            )

        shas: list[str] = []
        image_vectors: list[Sequence[float]] | None = None
        if result.images:
            shas = list(dict.fromkeys(image.sha256 for image in result.images))
            try:
                image_vectors = await self._embedder.embed_images(
                    [prepared.prepared_images[digest] for digest in shas]
                )
            except RemoteModelError as exc:
                logger.warning(
                    "图片向量不可用，视觉召回降级",
                    extra={"images": len(result.images), "error": type(exc).__name__},
                )

        if text_vectors is not None:
            await asyncio.to_thread(
                self._repo.write_text_vectors,
                user_id=user_id,
                vectors={
                    stored.unit_rowid: vector
                    for stored, vector in zip(result.units, text_vectors, strict=True)
                },
            )
        if image_vectors is not None:
            by_sha = dict(zip(shas, image_vectors, strict=True))
            await asyncio.to_thread(
                self._repo.write_image_vectors,
                user_id=user_id,
                vectors={
                    image.image_rowid: by_sha[image.sha256]
                    for image in result.images
                    if image.sha256 in by_sha
                },
            )

    # ---------------------------------------------------------------------- 检索

    async def retrieve(self, query: SearchQuery) -> Sequence[MemoryEntry]:
        text = _query_text(query)
        options = tuple(option for option in query.options if option.strip())
        raw_images = [part for part in query.parts if isinstance(part, ImageContent)][
            : max(0, self._settings.max_query_images)
        ]
        images = (
            await asyncio.to_thread(_prepare_parts, raw_images, self._settings)
            if raw_images
            else []
        )
        # 查询图片也走一次描述：图文混搜时，文字侧的查词与重排才拿到实义，
        # 否则「这张图是什么？」这类问句在重排阶段几乎是噪声。
        visual_text = " ".join(
            caption for caption in await self._caption_query(raw_images, images) if caption
        ).strip()
        search_text = "\n".join(part for part in (text, visual_text) if part)
        pool = max(self._settings.route_pool, query.top_k)

        if await self._should_abstain(query.user_id, text, visual_text, options):
            logger.info(
                "问题实词在语料中无命中，按弃答处理",
                extra={"user_id": query.user_id, "query_chars": len(text)},
            )
            return []

        rankings, route_stats = await self._recall(
            user_id=query.user_id,
            text=search_text,
            images=images,
            options=options,
            pool=pool,
        )
        if not rankings:
            return []
        candidate_count = len({item for ranking in rankings for item in ranking.ids})
        k = (
            adaptive_k(candidate_count, base=self._settings.rrf_k)
            if self._settings.rrf_k_adaptive
            else self._settings.rrf_k
        )
        # 正式评测 top_k=100，融合池要跟着放大，否则重排会先把候选截断到 60
        fuse_limit = max(self._settings.rerank_top_n, query.top_k)
        fused = reciprocal_rank_fusion(
            rankings,
            k=k,
            weights=self._settings.parsed_route_weights,
            limit=fuse_limit,
        )
        if not fused:
            return []

        stored = await asyncio.to_thread(self._repo.fetch_units, [hit.id for hit in fused])
        ordered = [stored[hit.id] for hit in fused if hit.id in stored]
        if not ordered:
            return []

        fused_scores = [hit.score for hit in fused if hit.id in stored]
        rerank_query = _rerank_query(search_text, options)
        scores, reranked = await self._rerank(rerank_query, ordered, fused_scores)
        if reranked:
            # 精排名次与召回名次再融合：cross-encoder 只决定邻近名次，多路共识仍然算数
            ranked = fuse_rerank_ranks(
                fused,
                scores,
                k=k,
                fusion_weight=self._settings.fusion_weight,
            )
            order = [index for index, _ in ranked]
            # score 用 cross-encoder 的绝对置信度（logit → sigmoid），按名次单调分配：
            # 归一化会把第一名永远压成 1.0，平台就没法用分数判断「这条到底有多相关」
            calibrated = sorted((sigmoid(value) for value in scores), reverse=True)
            final_scores = calibrated
        else:
            order = list(range(len(ordered)))
            final_scores = normalize_scores([fused_scores[index] for index in order])

        limit = query.top_k
        if (
            self._settings.weak_evidence_limit > 0
            and final_scores
            and final_scores[0] < self._settings.weak_evidence_threshold
        ):
            # 首条都几乎没有置信度时，一百条弱证据只会让答案模型硬凑一个答案
            limit = min(limit, self._settings.weak_evidence_limit)
        selected = [ordered[index] for index in order[:limit]]
        selected_scores = final_scores[:limit]
        response_images = await self._response_images(selected)
        session_labels = (
            await asyncio.to_thread(self._repo.session_labels, user_id=query.user_id)
            if self._settings.content_session_label
            else {}
        )
        neighbors = await asyncio.to_thread(
            self._repo.fetch_neighbor_units,
            anchor_rowids=[unit.unit_rowid for unit in selected],
            radius=self._settings.neighbor_expand,
        )

        logger.info(
            "检索完成",
            extra={
                "routes": route_stats,
                "candidates": candidate_count,
                "rrf_k": k,
                "fused": len(fused),
                "reranked": reranked,
                "returned": len(selected),
                "images": sum(len(items) for items in response_images.values()),
            },
        )
        return [
            MemoryEntry(
                id=unit.id,
                parts=_entry_parts(
                    unit,
                    response_images.get(unit.id, []),
                    self._settings,
                    session_labels.get(unit.session_id),
                    neighbors.get(unit.unit_rowid, []),
                ),
                score=score,
                created_at=_unit_time(unit),
            )
            for unit, score in zip(selected, selected_scores, strict=True)
        ]

    async def _should_abstain(
        self, user_id: str, text: str, visual_text: str, options: Sequence[str]
    ) -> bool:
        """问题里的名词/实体在语料里几乎都没出现过 → 返回空列表，而不是硬凑最像的。

        判据是「缺失质量」：每个名词的权重取 1/(df+1)，缺失词的质量占比超过阈值
        就认为这条记忆不存在。不存在的东西，交给平台答案模型弃答才是对的；
        返回 100 条无关聊天记录只会诱导它编一个答案。

        题目只有虚词时（「那个地方怎么样」）才把选项也算进来——这时候选项才是实体来源。
        """
        if not self._settings.abstain_enabled:
            return False
        # 英文题不弃答：改写（city → Lisbon）太常见，词面缺失容易被误判；
        # 中文对话里问的实体通常原样出现，缺失就真是没有这条记忆。
        cjk = _is_cjk_question(" ".join(part for part in (text, visual_text) if part))
        if not cjk:
            return False
        if await asyncio.to_thread(self._repo.has_image_units, user_id=user_id) and not cjk:
            return False
        question = "\n".join(part for part in (text, visual_text) if part)
        terms = salient_terms(question)
        minimum = max(1, self._settings.abstain_min_terms)
        if len(terms) < minimum:
            probe = "\n".join(part for part in (question, *options) if part)
            terms = salient_terms(probe)
        if len(terms) < minimum:
            return False
        threshold = self._settings.abstain_mass_ratio
        if await self._missing_ratio(user_id, terms) < threshold:
            return False
        # 题目本身没有证据，但选项里可能带着实体（「那个地方怎么样？」）
        option_terms = salient_terms("\n".join(options)) if options else []
        options_have_evidence = (
            len(option_terms) >= minimum
            and await self._missing_ratio(user_id, option_terms) < threshold
        )
        return not options_have_evidence

    async def _missing_ratio(self, user_id: str, terms: Sequence[str]) -> float:
        """名词的「缺失质量」占比：权重取 1/(df+1)，全缺失时返回 1.0。"""
        frequencies = await asyncio.to_thread(
            self._repo.term_frequencies, user_id=user_id, terms=terms
        )
        missing_mass = 0.0
        total_mass = 0.0
        for count in frequencies.values():
            weight = 1.0 / (count + 1)
            total_mass += weight
            if count == 0:
                missing_mass += weight
        return missing_mass / total_mass if total_mass > 0 else 0.0

    async def _caption_query(
        self, raw: Sequence[ImageContent], prepared: Sequence[PreparedImage]
    ) -> list[str]:
        """按原始字节的 sha256 查描述：与写入时同一把钥匙，同一张图只描述一次。"""
        if not raw:
            return []
        shas = [sha256_hex(item.data) for item in raw]
        captions = await self._resolve_captions(
            {sha: image for sha, image in zip(shas, prepared, strict=True)}
        )
        return [captions[sha].description if sha in captions else "" for sha in shas]

    async def _recall(
        self,
        *,
        user_id: str,
        text: str,
        images: list[PreparedImage],
        options: Sequence[str],
        pool: int,
    ) -> tuple[list[RouteRanking], dict[str, int]]:
        rankings: list[RouteRanking] = []
        stats: dict[str, int] = {}

        if text:
            lexical_routes = await asyncio.to_thread(
                self._repo.search_lexical,
                user_id=user_id,
                word_query=word_query(text),
                trigram_query=trigram_query(text),
                zh_query=zh_query(text),
                limit=pool,
            )
            for name, ids in lexical_routes:
                rankings.append(RouteRanking(name=name, ids=ids))
                stats[name] = len(ids)

        option_ids = await self._option_route(user_id, options)
        if option_ids:
            rankings.append(RouteRanking(name="options_lexical", ids=option_ids))
            stats["options_lexical"] = len(option_ids)

        embed_text = clip_middle(text, self._settings.embedding_max_query_chars)
        vectors: list[tuple[str, Sequence[float]]] = []
        try:
            if embed_text:
                vectors.append(("query", (await self._embedder.embed_texts([embed_text]))[0]))
            if images:
                image_vectors = await self._embedder.embed_images(images)
                for index, vector in enumerate(image_vectors):
                    vectors.append((f"image{index}", vector))
                if embed_text:
                    vectors.append(
                        (
                            "query_image",
                            await self._embedder.embed_multimodal(text=embed_text, images=images),
                        )
                    )
        except RemoteModelError as exc:
            logger.warning("查询向量不可用，退到词面召回", extra={"error": type(exc).__name__})

        for label, vector in vectors:
            for kind in ("text", "image"):
                ids = await asyncio.to_thread(
                    self._repo.search_vectors,
                    kind=kind,
                    user_id=user_id,
                    vector=vector,
                    limit=pool,
                )
                if not ids:
                    continue
                name = _route_name(kind, label)
                rankings.append(RouteRanking(name=name, ids=ids))
                stats[name] = len(ids)
        return rankings, stats

    async def _option_route(self, user_id: str, options: Sequence[str]) -> tuple[str, ...]:
        """按选项逐条召回：选项本身就是最接近答案的查询。

        每个选项只取少数几条，再按选项顺序轮流合并，避免某个选项的候选占满整条路。
        """
        if not options or not self._settings.options_enabled:
            return ()
        unique = list(
            dict.fromkeys(
                strip_option_label(option) for option in options if strip_option_label(option)
            )
        )[: self._settings.max_options]
        if not unique:
            return ()
        per_option = max(1, self._settings.options_per_option)
        merged: list[str] = []
        seen: set[str] = set()
        branches: list[list[str]] = []
        for option in unique:
            routes = await asyncio.to_thread(
                self._repo.search_lexical,
                user_id=user_id,
                word_query=word_query(option),
                trigram_query=trigram_query(option),
                zh_query=zh_query(option),
                limit=per_option,
            )
            branch: list[str] = []
            for _name, ids in routes:
                for item in ids:
                    if item not in branch:
                        branch.append(item)
            branches.append(branch[:per_option])
        for depth in range(per_option):
            for branch in branches:
                if depth < len(branch) and branch[depth] not in seen:
                    seen.add(branch[depth])
                    merged.append(branch[depth])
        return tuple(merged)

    async def _rerank(
        self, text: str, units: list[StoredUnit], fallback: Sequence[float]
    ) -> tuple[list[float], bool]:
        if not self._settings.rerank_enabled or not units or not text.strip():
            return list(fallback), False
        query = clip_middle(text, self._settings.rerank_max_query_chars)
        documents = [truncate(unit.lexical_full, self._settings.rerank_doc_chars) for unit in units]
        try:
            scores = await self._reranker.rerank(query=query, documents=documents)
        except RemoteModelError as exc:
            logger.warning("重排不可用，按融合分返回", extra={"error": type(exc).__name__})
            return list(fallback), False
        return list(scores), True

    async def _response_images(self, units: Sequence[StoredUnit]) -> dict[str, list[ImageContent]]:
        """把命中记忆里的原图回传给平台：图里的字，描述未必抄全，像素才是原始证据。

        只按名次顺序配额：先满足前几条，再受总字节数约束（契约上限 30 MiB/响应）。
        """
        if not self._settings.return_images:
            return {}
        ids = [unit.id for unit in units if unit.image_count]
        if not ids:
            return {}
        stored = await asyncio.to_thread(self._repo.fetch_images, ids)
        ordered_images: list[MemoryImage] = [
            image for unit in units for image in stored.get(unit.id, [])
        ]
        if not ordered_images:
            return {}
        prepared = await asyncio.to_thread(
            self._prepare_response_images,
            ordered_images[: self._settings.max_returned_images],
        )
        budget = self._settings.max_response_image_bytes
        by_unit: dict[str, list[ImageContent]] = {}
        index = 0
        for unit in units:
            images = stored.get(unit.id, [])
            for _ in images:
                if index >= len(prepared) or budget <= 0:
                    break
                item = prepared[index]
                index += 1
                budget -= len(item.data)
                by_unit.setdefault(unit.id, []).append(
                    ImageContent(media_type=item.media_type, data=item.data)
                )
        return by_unit

    def _prepare_response_images(self, images: Sequence[MemoryImage]) -> list[PreparedImage]:
        """小图原样回传（保留可读性），大图才缩放；同一张图缓存只算一次。"""
        prepared: list[PreparedImage] = []
        for image in images:
            cached = self._image_cache.get(image.sha256)
            if cached is None:
                if len(image.data) <= self._settings.response_image_passthrough_bytes:
                    cached = PreparedImage(
                        media_type=image.media_type,
                        data=image.data,
                        width=0,
                        height=0,
                        resized=False,
                    )
                else:
                    cached = prepare_for_model(
                        InlineImage(media_type=image.media_type, data=image.data),
                        max_side=self._settings.response_image_max_side,
                        quality=self._settings.response_image_quality,
                    )
                if len(self._image_cache) >= 512:
                    self._image_cache.clear()
                self._image_cache[image.sha256] = cached
            prepared.append(cached)
        return prepared


def _prepare_occurrences(
    parts: Sequence[ImageContent], settings: Settings
) -> list[tuple[str, PreparedImage]]:
    return [
        (sha256_hex(part.data), prepare_for_model(part, max_side=settings.image_max_side))
        for part in parts
    ]


def _prepare_parts(parts: Sequence[ImageContent], settings: Settings) -> list[PreparedImage]:
    return [prepare_for_model(part, max_side=settings.image_max_side) for part in parts]


def _unit_id(user_id: str, session_id: str, request_id: str, ordinal: int) -> str:
    raw = f"{user_id}\x1f{session_id}\x1f{request_id}\x1f{ordinal}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"{ID_PREFIX}{digest[:ID_DIGEST_CHARS]}"


def _message_text(message: Message) -> str:
    return "\n".join(
        part.text.strip()
        for part in message.parts
        if isinstance(part, TextContent) and part.text.strip()
    )


def _query_text(query: SearchQuery) -> str:
    return "\n".join(
        part.text.strip()
        for part in query.parts
        if isinstance(part, TextContent) and part.text.strip()
    )


def _is_cjk_question(text: str) -> bool:
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return letters > 0 and cjk / letters >= 0.3


def _rerank_query(text: str, options: Sequence[str]) -> str:
    if not options:
        return text
    labelled = "\n".join(option.strip() for option in options if option.strip())
    return f"{text}\n\nOptions:\n{labelled}" if text else f"Options:\n{labelled}"


def _route_name(kind: str, label: str) -> str:
    if label == "query":
        return f"dense_{kind}"
    if label == "query_image":
        return f"dense_{kind}_multimodal"
    return f"dense_{kind}_{label}"


def _entry_parts(
    unit: StoredUnit,
    images: Sequence[ImageContent],
    settings: Settings,
    session_label: str | None = None,
    neighbors: Sequence[StoredUnit] = (),
) -> tuple[Content, ...]:
    text = render_text(
        role=unit.role,
        timestamp_ms=unit.ts_ms,
        text=unit.text,
        captions=unit.captions,
        transcripts=unit.transcripts,
        include_timestamp=settings.content_include_timestamp,
        include_role=settings.content_include_role,
        session_label=session_label,
    )
    context = _neighbor_text(neighbors, budget=settings.neighbor_expand_chars)
    if context:
        text = f"{text}\n{context}"
    parts: list[Content] = [TextContent(text=text)]
    parts.extend(images)
    return tuple(parts)


def _neighbor_text(neighbors: Sequence[StoredUnit], *, budget: int) -> str:
    """相邻几轮拼成一段上下文；超过预算就停，避免挤掉别的证据。"""
    lines: list[str] = []
    used = 0
    for neighbor in neighbors:
        body = neighbor.text.strip()
        if not body and neighbor.captions:
            body = " ".join(f"[image] {caption}" for caption in neighbor.captions)
        if not body:
            continue
        label = "user" if neighbor.role.value == "user" else "assistant"
        line = f"[邻近消息] {label}: {body}"
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def _unit_time(unit: StoredUnit) -> datetime:
    if unit.ts_ms is None:
        return unit.created_at
    try:
        return datetime.fromtimestamp(unit.ts_ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return unit.created_at
