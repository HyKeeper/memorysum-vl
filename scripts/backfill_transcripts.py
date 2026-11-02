#!/usr/bin/env python3
"""给库里已有的图片补「逐条转写」，并重建这些记忆的文本索引。

为什么需要它：评测数据里的图片（聊天截图/登机牌/文档）答案常常只在像素里，
描述抄不到细节；补一段逐条转写后，词法/重排才有可用的文字证据。
只重算文本列与 FTS，**不重算向量**（embedding 只吃描述，不吃转写）。

```
.venv/bin/python scripts/backfill_transcripts.py            # 只处理缺转写的图
.venv/bin/python scripts/backfill_transcripts.py --dry-run  # 看看有多少张
.venv/bin/python scripts/backfill_transcripts.py --redo     # 连已有的也重做
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memorysum_vl.adapters.llm.caption import RaxCaptioner  # noqa: E402
from memorysum_vl.adapters.llm.embedding import QwenVLEmbedder  # noqa: E402
from memorysum_vl.adapters.stores.repository import (  # noqa: E402
    CaptionRecord,
    SqliteRepository,
)
from memorysum_vl.core.config import Settings  # noqa: E402
from memorysum_vl.core.imaging import prepare_for_model  # noqa: E402
from memorysum_vl.core.media import InlineImage  # noqa: E402
from memorysum_vl.core.render import render_text  # noqa: E402
from memorysum_vl.core.text import zh_index_text  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--redo", action="store_true", help="已有转写的也重做")
    parser.add_argument("--user", default="", help="只处理某个 user_id")
    parser.add_argument(
        "--missing-caption",
        action="store_true",
        help="只补描述/转写实际为空的单元（不再只信 caption_state）",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="描述补齐后重算这些单元的文本向量（vec_text 吃 lexical，含描述）",
    )
    args = parser.parse_args()

    settings = Settings()
    repository = SqliteRepository(settings.db_path, embedding_dim=settings.embedding_dim)
    captioner = RaxCaptioner(settings)
    try:
        rows = repository._conn.execute(
            "SELECT mi.sha256, mi.media_type, mi.data, mi.ordinal, u.unit_rowid, u.id AS unit_id,"
            " u.user_id, u.text, u.captions, u.transcripts, u.role, u.ts_ms"
            ", u.caption_state"
            " FROM memory_image mi JOIN memory_unit u ON u.id = mi.memory_id"
            + (" WHERE u.user_id = ?" if args.user else "")
            + " ORDER BY u.unit_rowid, mi.ordinal",
            (args.user,) if args.user else (),
        ).fetchall()
        if not rows:
            print("库里没有图片")
            return 0

        units: dict[int, dict] = {}
        for row in rows:
            unit = units.setdefault(
                int(row["unit_rowid"]),
                {
                    "unit_rowid": int(row["unit_rowid"]),
                    "id": row["unit_id"],
                    "user_id": row["user_id"],
                    "text": row["text"],
                    "role": row["role"],
                    "ts_ms": row["ts_ms"],
                    "caption_state": row["caption_state"],
                    "captions": list(json.loads(row["captions"] or "[]")),
                    "transcripts": list(json.loads(row["transcripts"] or "[]")),
                    "images": [],
                },
            )
            unit["images"].append(row)

        def missing(unit: dict) -> bool:
            if args.redo:
                return True
            # caption_state 会说谎：抖动窗口里丢了描述的单元被上一轮回填标成了 'ok'，
            # 于是 --missing-caption 再也扫不到它们。这里直接在数据上核对是否真的为空。
            blank_caption = len(unit["captions"]) < len(unit["images"]) or any(
                not str(value).strip() for value in unit["captions"]
            )
            if args.missing_caption:
                return unit["caption_state"] == "missing" or blank_caption
            if unit["caption_state"] == "missing" or blank_caption:
                return True
            for index in range(len(unit["images"])):
                stored = unit["transcripts"][index] if index < len(unit["transcripts"]) else ""
                if not stored.strip():
                    return True
            return False

        pending = [unit for unit in units.values() if missing(unit)]
        print(f"图片记忆 {len(units)} 条，需要补转写 {len(pending)} 条")
        if args.dry_run or not pending:
            return 0

        updates = []
        caption_fixes: list[tuple[str, int, str]] = []
        for unit in pending:
            transcripts = list(unit["transcripts"])
            captions = list(unit["captions"])
            records: list[CaptionRecord] = []
            for index, image in enumerate(unit["images"]):
                inline = InlineImage(media_type=image["media_type"], data=image["data"])
                prepared = prepare_for_model(inline, max_side=settings.image_max_side)
                try:
                    result = (await captioner.caption([prepared]))[0]
                except Exception as exc:  # noqa: BLE001 - 单张图失败不该中断整批
                    print(f"  跳过 {unit['id']} 第 {index} 张：{type(exc).__name__}")
                    continue
                while len(transcripts) <= index:
                    transcripts.append("")
                if result.transcript:
                    transcripts[index] = result.transcript
                if index >= len(captions):
                    captions.append(result.description)
                elif not str(captions[index]).strip() and result.description.strip():
                    # 只判 len() 会让 [""] 这种空占位把新描述吃掉，等于永远补不上
                    captions[index] = result.description
                if str(captions[index]).strip():
                    caption_fixes.append((unit["id"], int(image["ordinal"]), captions[index]))
                records.append(
                    CaptionRecord(
                        sha256=image["sha256"],
                        media_type=image["media_type"],
                        caption=captions[index],
                        transcript=transcripts[index],
                    )
                )
            repository.store_captions(records, model=settings.caption_model)
            lexical = render_text(
                role=unit["role"],
                timestamp_ms=unit["ts_ms"],
                text=unit["text"],
                captions=tuple(captions),
                include_transcripts=False,
            )
            lexical_full = render_text(
                role=unit["role"],
                timestamp_ms=unit["ts_ms"],
                text=unit["text"],
                captions=tuple(captions),
                transcripts=tuple(transcripts),
            )
            updates.append(
                SqliteRepository.ReindexUpdate(
                    unit_rowid=unit["unit_rowid"],
                    unit_id=unit["id"],
                    user_id=unit["user_id"],
                    captions=tuple(captions),
                    transcripts=tuple(transcripts),
                    lexical=lexical,
                    lexical_full=lexical_full,
                    lexical_zh=zh_index_text(lexical_full),
                )
            )
            print(f"  {unit['id']} 转写 {sum(1 for t in transcripts if t)} 张")
        repository.reindex_units(updates)
        if updates:
            with repository._lock, repository._conn:
                repository._conn.executemany(
                    "UPDATE memory_unit SET caption_state = 'ok' WHERE unit_rowid = ?",
                    [(update.unit_rowid,) for update in updates],
                )
        print(f"已重算 {len(updates)} 条记忆的文本索引")

        # memory_image.caption 是 Add 时写一次的冗余列，回填历来不同步，
        # 会留下「image_caption 有描述、memory_image 却是空」的自相矛盾状态。
        if caption_fixes:
            with repository._lock, repository._conn:
                repository._conn.executemany(
                    "UPDATE memory_image SET caption = ? WHERE memory_id = ? AND ordinal = ?",
                    [(cap, mid, ord_) for mid, ord_, cap in caption_fixes],
                )
            print(f"已同步 {len(caption_fixes)} 张图片的冗余描述列")

        if args.reembed and updates:
            embedder = QwenVLEmbedder(settings)
            try:
                by_rowid = {update.unit_rowid: update for update in updates}
                for user_id in sorted({update.user_id for update in updates}):
                    rowids = [rowid for rowid, u in by_rowid.items() if u.user_id == user_id]
                    vectors = await embedder.embed_texts([by_rowid[r].lexical for r in rowids])
                    await asyncio.to_thread(
                        repository.write_text_vectors,
                        user_id=user_id,
                        vectors=dict(zip(rowids, vectors, strict=True)),
                    )
                print(f"已重算 {len(updates)} 条记忆的文本向量")
            finally:
                await embedder.aclose()
        elif updates:
            print("提示：未加 --reembed，vec_text 仍是旧向量（描述只进了词面索引）")
        return 0
    finally:
        await captioner.aclose()
        repository.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
