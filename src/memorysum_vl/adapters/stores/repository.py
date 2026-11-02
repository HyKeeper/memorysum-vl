"""SQLite 持久层：原文、图片字节、FTS5 词面索引、sqlite-vec 向量索引。

一个库文件承担四种角色：

| 表 | 用途 |
| --- | --- |
| `memory_unit` | 一条消息一行，保留原文、图片描述与可检索文本 |
| `memory_image` | 原图字节，按 `(memory_id, ordinal)` 对齐消息内的图片顺序 |
| `image_caption` | 按 sha256 缓存描述，同一张图重复出现只描述一次 |
| `fts_word` / `fts_trigram` | 两套词面索引：分词器互补，中文靠二元组补 |
| `vec_text` / `vec_image` | 文本与图片各自的向量，`user_id` 作 partition key |

写路径全部走显式事务：一次 Add 要么整体可见，要么整体不可见。重发同一个
`request_id` 会先删掉该请求写入过的行再重写，保证幂等。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import sqlite_vec

from memorysum_vl.core.logging import get_logger
from memorysum_vl.core.text import (
    camel_expansions,
    expand_for_word_index,
    zh_index_text,
)
from memorysum_vl.domain.models import MemoryImage, MemoryUnit, Role

logger = get_logger(__name__)

SCHEMA_VERSION: Final[str] = "2"
VectorKind = Literal["text", "image"]
_MAX_IDS_PER_QUERY: Final[int] = 400


@dataclass(frozen=True, slots=True)
class StoredUnit:
    unit_rowid: int
    id: str
    user_id: str
    session_id: str
    request_id: str
    ordinal: int
    role: Role
    ts_ms: int | None
    created_at: datetime
    text: str
    captions: tuple[str, ...]
    transcripts: tuple[str, ...]
    lexical: str
    lexical_full: str
    lexical_zh: str
    image_count: int


@dataclass(frozen=True, slots=True)
class StoredImage:
    image_rowid: int
    memory_id: str
    ordinal: int
    sha256: str


@dataclass(frozen=True, slots=True)
class WriteResult:
    units: tuple[StoredUnit, ...]
    images: tuple[StoredImage, ...]


@dataclass(frozen=True, slots=True)
class CaptionRecord:
    sha256: str
    media_type: str
    caption: str
    transcript: str = ""


@dataclass(frozen=True, slots=True)
class CachedCaption:
    description: str
    transcript: str


class SqliteRepository:
    def __init__(self, path: Path, *, embedding_dim: int) -> None:
        self._path = path
        self._dim = embedding_dim
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()

    # ------------------------------------------------------------------ 生命周期

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def states(self, *, user_id: str) -> list[tuple[str, str]]:
        """每条记忆的 (caption_state, embedding_state)，降级路径的断言用。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT caption_state, embedding_state FROM memory_unit"
                " WHERE user_id = ? ORDER BY unit_rowid",
                (user_id,),
            ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def count_units(self, *, user_id: str | None = None) -> int:
        with self._lock:
            if user_id is None:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM memory_unit").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM memory_unit WHERE user_id = ?", (user_id,)
                ).fetchone()
        return int(row["n"])

    # ---------------------------------------------------------------------- 写入

    def replace_request(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        units: Sequence[MemoryUnit],
        lexicals: Sequence[str],
        lexicals_full: Sequence[str],
        lexicals_zh: Sequence[str],
    ) -> WriteResult:
        """同一 (user, session, request) 的行先清空再写入，重放不会重复计数。"""
        if not (len(units) == len(lexicals) == len(lexicals_full) == len(lexicals_zh)):
            raise ValueError("units / lexicals / lexicals_full / lexicals_zh must be aligned")
        with self._lock, _transaction(self._conn) as cursor:
            cursor.execute(
                "SELECT unit_rowid, id FROM memory_unit "
                "WHERE user_id = ? AND session_id = ? AND request_id = ?",
                (user_id, session_id, request_id),
            )
            stale = cursor.fetchall()
            if stale:
                cursor.executemany(
                    "DELETE FROM vec_text WHERE unit_rowid = ?",
                    [(row["unit_rowid"],) for row in stale],
                )
                cursor.executemany(
                    "DELETE FROM fts_word WHERE rowid = ?",
                    [(row["unit_rowid"],) for row in stale],
                )
                cursor.executemany(
                    "DELETE FROM fts_trigram WHERE rowid = ?",
                    [(row["unit_rowid"],) for row in stale],
                )
                cursor.executemany(
                    "DELETE FROM fts_zh WHERE rowid = ?",
                    [(row["unit_rowid"],) for row in stale],
                )
                cursor.executemany(
                    "DELETE FROM memory_unit WHERE id = ?", [(row["id"],) for row in stale]
                )

            stored_units: list[StoredUnit] = []
            stored_images: list[StoredImage] = []
            for unit, lexical, lexical_full, lexical_zh in zip(
                units, lexicals, lexicals_full, lexicals_zh, strict=True
            ):
                cursor.execute(
                    "INSERT INTO memory_unit ("
                    " id, user_id, session_id, request_id, ordinal, role, ts_ms, created_at,"
                    " text, captions, transcripts, lexical, lexical_full, lexical_zh,"
                    " image_count, caption_state, embedding_state"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        unit.id,
                        unit.user_id,
                        unit.session_id,
                        unit.request_id,
                        unit.ordinal,
                        unit.role.value,
                        unit.timestamp_ms,
                        unit.created_at.isoformat(),
                        unit.text,
                        json.dumps(list(unit.captions), ensure_ascii=False),
                        json.dumps(list(unit.transcripts), ensure_ascii=False),
                        lexical,
                        lexical_full,
                        lexical_zh,
                        len(unit.images),
                        unit.caption_state,
                        unit.embedding_state,
                    ),
                )
                unit_rowid = int(cursor.lastrowid)
                camel = camel_expansions(lexical_full)
                cursor.execute(
                    "INSERT INTO fts_word (rowid, memory_id, user_id, lexical) VALUES (?, ?, ?, ?)",
                    (
                        unit_rowid,
                        unit.id,
                        unit.user_id,
                        f"{expand_for_word_index(lexical_full)} {camel}"
                        if camel
                        else expand_for_word_index(lexical_full),
                    ),
                )
                cursor.execute(
                    "INSERT INTO fts_zh (rowid, memory_id, user_id, lexical_zh)"
                    " VALUES (?, ?, ?, ?)",
                    (unit_rowid, unit.id, unit.user_id, lexical_zh),
                )
                cursor.execute(
                    "INSERT INTO fts_trigram (rowid, memory_id, user_id, lexical) "
                    "VALUES (?, ?, ?, ?)",
                    (unit_rowid, unit.id, unit.user_id, lexical_full),
                )
                for ordinal, image in enumerate(unit.images):
                    cursor.execute(
                        "INSERT INTO memory_image"
                        " (memory_id, ordinal, media_type, sha256, caption, data)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            unit.id,
                            ordinal,
                            image.media_type,
                            image.sha256,
                            image.caption,
                            image.data,
                        ),
                    )
                    stored_images.append(
                        StoredImage(
                            image_rowid=int(cursor.lastrowid),
                            memory_id=unit.id,
                            ordinal=ordinal,
                            sha256=image.sha256,
                        )
                    )
                stored_units.append(
                    StoredUnit(
                        unit_rowid=unit_rowid,
                        id=unit.id,
                        user_id=unit.user_id,
                        session_id=unit.session_id,
                        request_id=unit.request_id,
                        ordinal=unit.ordinal,
                        role=unit.role,
                        ts_ms=unit.timestamp_ms,
                        created_at=unit.created_at,
                        text=unit.text,
                        captions=unit.captions,
                        transcripts=unit.transcripts,
                        lexical=lexical,
                        lexical_full=lexical_full,
                        lexical_zh=lexical_zh,
                        image_count=len(unit.images),
                    )
                )
        return WriteResult(units=tuple(stored_units), images=tuple(stored_images))

    def write_text_vectors(self, *, user_id: str, vectors: Mapping[int, Sequence[float]]) -> None:
        """unit_rowid → 向量；写前先删旧值，避免重复 Add 留下两份。"""
        if not vectors:
            return
        rows = [
            (rowid, user_id, sqlite_vec.serialize_float32(list(vector)))
            for rowid, vector in vectors.items()
        ]
        with self._lock, _transaction(self._conn) as cursor:
            cursor.executemany(
                "DELETE FROM vec_text WHERE unit_rowid = ?", [(row[0],) for row in rows]
            )
            cursor.executemany(
                "INSERT INTO vec_text (unit_rowid, user_id, embedding) VALUES (?, ?, ?)", rows
            )

    def write_image_vectors(self, *, user_id: str, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        rows = [
            (rowid, user_id, sqlite_vec.serialize_float32(list(vector)))
            for rowid, vector in vectors.items()
        ]
        with self._lock, _transaction(self._conn) as cursor:
            cursor.executemany(
                "DELETE FROM vec_image WHERE image_rowid = ?", [(row[0],) for row in rows]
            )
            cursor.executemany(
                "INSERT INTO vec_image (image_rowid, user_id, embedding) VALUES (?, ?, ?)", rows
            )

    @dataclass(frozen=True, slots=True)
    class ReindexUpdate:
        unit_rowid: int
        unit_id: str
        user_id: str
        captions: Sequence[str]
        transcripts: Sequence[str]
        lexical: str
        lexical_full: str
        lexical_zh: str

    def reindex_units(self, updates: Sequence[SqliteRepository.ReindexUpdate]) -> None:
        """重算文本列与三张 FTS 表；向量不动（embedding 只吃描述，不吃转写）。"""
        if not updates:
            return
        with self._lock, _transaction(self._conn) as cursor:
            for update in updates:
                cursor.execute(
                    "UPDATE memory_unit SET captions = ?, transcripts = ?, lexical = ?,"
                    " lexical_full = ?, lexical_zh = ? WHERE unit_rowid = ?",
                    (
                        json.dumps(list(update.captions), ensure_ascii=False),
                        json.dumps(list(update.transcripts), ensure_ascii=False),
                        update.lexical,
                        update.lexical_full,
                        update.lexical_zh,
                        update.unit_rowid,
                    ),
                )
                camel = camel_expansions(update.lexical_full)
                word_text = expand_for_word_index(update.lexical_full)
                cursor.execute("DELETE FROM fts_word WHERE rowid = ?", (update.unit_rowid,))
                cursor.execute("DELETE FROM fts_trigram WHERE rowid = ?", (update.unit_rowid,))
                cursor.execute("DELETE FROM fts_zh WHERE rowid = ?", (update.unit_rowid,))
                cursor.execute(
                    "INSERT INTO fts_word (rowid, memory_id, user_id, lexical) VALUES (?, ?, ?, ?)",
                    (
                        update.unit_rowid,
                        update.unit_id,
                        update.user_id,
                        f"{word_text} {camel}" if camel else word_text,
                    ),
                )
                cursor.execute(
                    "INSERT INTO fts_trigram (rowid, memory_id, user_id, lexical)"
                    " VALUES (?, ?, ?, ?)",
                    (update.unit_rowid, update.unit_id, update.user_id, update.lexical_full),
                )
                cursor.execute(
                    "INSERT INTO fts_zh (rowid, memory_id, user_id, lexical_zh)"
                    " VALUES (?, ?, ?, ?)",
                    (update.unit_rowid, update.unit_id, update.user_id, update.lexical_zh),
                )

    def mark_states(self, *, unit_ids: Sequence[str], column: str, state: str) -> None:
        if column not in {"embedding_state", "caption_state"}:
            raise ValueError(f"unsupported state column {column!r}")
        if not unit_ids:
            return
        with self._lock, _transaction(self._conn) as cursor:
            cursor.executemany(
                f"UPDATE memory_unit SET {column} = ? WHERE id = ?",
                [(state, unit_id) for unit_id in unit_ids],
            )

    # ---------------------------------------------------------------------- 描述缓存

    def lookup_captions(self, shas: Sequence[str]) -> dict[str, CachedCaption]:
        unique = list(dict.fromkeys(shas))
        if not unique:
            return {}
        found: dict[str, CachedCaption] = {}
        with self._lock:
            for chunk in _chunks(unique, _MAX_IDS_PER_QUERY):
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    "SELECT sha256, caption, transcript FROM image_caption"
                    f" WHERE sha256 IN ({placeholders})",
                    tuple(chunk),
                ).fetchall()
                found.update(
                    {
                        row["sha256"]: CachedCaption(
                            description=str(row["caption"]),
                            transcript=str(row["transcript"] or ""),
                        )
                        for row in rows
                    }
                )
        return found

    def store_captions(self, records: Sequence[CaptionRecord], *, model: str) -> None:
        if not records:
            return
        now = datetime.now(tz=UTC).isoformat()
        with self._lock, _transaction(self._conn) as cursor:
            cursor.executemany(
                "INSERT INTO image_caption"
                " (sha256, media_type, model, caption, transcript, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(sha256) DO UPDATE SET"
                " caption = excluded.caption, transcript = excluded.transcript,"
                " model = excluded.model",
                [
                    (
                        record.sha256,
                        record.media_type,
                        model,
                        record.caption,
                        record.transcript,
                        now,
                    )
                    for record in records
                ],
            )

    # ---------------------------------------------------------------------- 召回

    def search_lexical(
        self,
        *,
        user_id: str,
        word_query: str | None,
        trigram_query: str | None,
        zh_query: str | None = None,
        limit: int = 100,
    ) -> list[tuple[str, tuple[str, ...]]]:
        """返回 [(路名, memory_id 有序元组)]：英文词 / 中文词 / 三元组各算一路。"""
        routes: list[tuple[str, tuple[str, ...]]] = []
        with self._lock:
            if zh_query:
                rows = self._conn.execute(
                    "SELECT memory_id FROM fts_zh WHERE fts_zh MATCH ? AND user_id = ? "
                    "ORDER BY bm25(fts_zh), memory_id LIMIT ?",
                    (zh_query, user_id, limit),
                ).fetchall()
                routes.append(("lexical_zh", tuple(row["memory_id"] for row in rows)))
            if word_query:
                rows = self._conn.execute(
                    "SELECT memory_id FROM fts_word WHERE fts_word MATCH ? AND user_id = ? "
                    "ORDER BY bm25(fts_word), memory_id LIMIT ?",
                    (word_query, user_id, limit),
                ).fetchall()
                routes.append(("lexical", tuple(row["memory_id"] for row in rows)))
            if trigram_query:
                rows = self._conn.execute(
                    "SELECT memory_id FROM fts_trigram WHERE fts_trigram MATCH ? AND user_id = ? "
                    "ORDER BY bm25(fts_trigram), memory_id LIMIT ?",
                    (trigram_query, user_id, limit),
                ).fetchall()
                routes.append(("lexical_ngram", tuple(row["memory_id"] for row in rows)))
        return routes

    def search_vectors(
        self,
        *,
        kind: VectorKind,
        user_id: str,
        vector: Sequence[float],
        limit: int,
    ) -> tuple[str, ...]:
        """KNN 召回，按 memory_id 去重（同一单元的多张图只保留最近的那次命中）。"""
        blob = sqlite_vec.serialize_float32(list(vector))
        with self._lock:
            if kind == "text":
                rows = self._conn.execute(
                    "SELECT u.id AS id FROM ("
                    "  SELECT unit_rowid FROM vec_text"
                    "  WHERE embedding MATCH ? AND k = ? AND user_id = ?"
                    ") v JOIN memory_unit u ON u.unit_rowid = v.unit_rowid",
                    (blob, limit, user_id),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT u.id AS id FROM ("
                    "  SELECT image_rowid FROM vec_image"
                    "  WHERE embedding MATCH ? AND k = ? AND user_id = ?"
                    ") v JOIN memory_image mi ON mi.image_rowid = v.image_rowid"
                    " JOIN memory_unit u ON u.id = mi.memory_id",
                    (blob, limit, user_id),
                ).fetchall()
        return tuple(dict.fromkeys(row["id"] for row in rows))

    def term_frequencies(self, *, user_id: str, terms: Sequence[str]) -> dict[str, int]:
        """每个实词在用户语料里出现在多少条记忆里（含原文/描述/转写）。"""
        counts: dict[str, int] = {}
        with self._lock:
            for term in terms:
                escaped = (
                    term.replace(chr(92), chr(92) * 2)
                    .replace("%", chr(92) + "%")
                    .replace("_", chr(92) + "_")
                )
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM memory_unit WHERE user_id = ?"
                    " AND lexical_full LIKE ? ESCAPE '\\'",
                    (user_id, f"%{escaped}%"),
                ).fetchone()
                counts[term] = int(row["n"]) if row else 0
        return counts

    def matched_terms(self, *, user_id: str, terms: Sequence[str]) -> list[str]:
        """返回在语料原文里出现过（子串命中）的实词列表。

        这里刻意用 LIKE 而不是 FTS：中文单字（狗/猫/车）在 FTS 里没有独立 token，
        而弃答判定必须能看到它们。
        """
        matched: list[str] = []
        with self._lock:
            for term in terms:
                escaped = (
                    term.replace(chr(92), chr(92) * 2)
                    .replace("%", chr(92) + "%")
                    .replace("_", chr(92) + "_")
                )
                row = self._conn.execute(
                    "SELECT 1 FROM memory_unit WHERE user_id = ?"
                    " AND lexical_full LIKE ? ESCAPE '\\' LIMIT 1",
                    (user_id, f"%{escaped}%"),
                ).fetchone()
                if row is not None:
                    matched.append(term)
        return matched

    def has_image_units(self, *, user_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM memory_unit WHERE user_id = ? AND image_count > 0 LIMIT 1",
                (user_id,),
            ).fetchone()
        return row is not None

    def session_labels(self, *, user_id: str) -> dict[str, str]:
        """给用户的所有会话按时间排序编号：`Session 3 (2024-05-06)`。

        会话 id 是哈希，对答案模型没有意义；平台原题里的「哪个会话」只能靠时间和
        序号表达。同一会话的所有消息拿到同一个标签，跨会话比较才有锚点。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, MIN(ts_ms) AS lo, MIN(created_at) AS created_at"
                " FROM memory_unit WHERE user_id = ? GROUP BY session_id"
                " ORDER BY COALESCE(MIN(ts_ms), 0), session_id",
                (user_id,),
            ).fetchall()
        labels: dict[str, str] = {}
        for index, row in enumerate(rows, start=1):
            moment = ""
            if row["lo"] is not None:
                try:
                    moment = datetime.fromtimestamp(int(row["lo"]) / 1000, tz=UTC).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                except (OverflowError, OSError, ValueError):
                    moment = ""
            if not moment:
                moment = str(row["created_at"])[:10]
            labels[str(row["session_id"])] = f"Session {index} ({moment})"
        return labels

    def fetch_units(self, ids: Sequence[str]) -> dict[str, StoredUnit]:
        unique = list(dict.fromkeys(ids))
        if not unique:
            return {}
        found: dict[str, StoredUnit] = {}
        with self._lock:
            for chunk in _chunks(unique, _MAX_IDS_PER_QUERY):
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    "SELECT * FROM memory_unit WHERE id IN (" + placeholders + ")",
                    tuple(chunk),
                ).fetchall()
                for row in rows:
                    found[row["id"]] = _row_to_unit(row)
        return found

    def fetch_neighbor_units(
        self, *, anchor_rowids: Sequence[int], radius: int
    ) -> dict[int, list[StoredUnit]]:
        """返回每个锚点单元同会话 ±radius 条邻居（不含自己），按 ordinal 排序。"""
        if not anchor_rowids or radius <= 0:
            return {}
        result: dict[int, list[StoredUnit]] = {}
        with self._lock:
            for chunk in _chunks([str(r) for r in anchor_rowids], _MAX_IDS_PER_QUERY):
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    "SELECT anchor.unit_rowid AS anchor_rowid, n.*"
                    " FROM memory_unit anchor"
                    " JOIN memory_unit n"
                    "   ON n.user_id = anchor.user_id AND n.session_id = anchor.session_id"
                    "  AND n.ordinal BETWEEN anchor.ordinal - ? AND anchor.ordinal + ?"
                    "  AND n.unit_rowid != anchor.unit_rowid"
                    f" WHERE anchor.unit_rowid IN ({placeholders})"
                    " ORDER BY anchor.unit_rowid, n.ordinal",
                    (radius, radius, *(int(item) for item in chunk)),
                ).fetchall()
                for row in rows:
                    result.setdefault(int(row["anchor_rowid"]), []).append(_row_to_unit(row))
        return result

    def fetch_images(self, memory_ids: Sequence[str]) -> dict[str, list[MemoryImage]]:
        unique = list(dict.fromkeys(memory_ids))
        if not unique:
            return {}
        found: dict[str, list[MemoryImage]] = {}
        with self._lock:
            for chunk in _chunks(unique, _MAX_IDS_PER_QUERY):
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    "SELECT memory_id, media_type, sha256, caption, data FROM memory_image "
                    "WHERE memory_id IN (" + placeholders + ") ORDER BY memory_id, ordinal",
                    tuple(chunk),
                ).fetchall()
                for row in rows:
                    found.setdefault(row["memory_id"], []).append(
                        MemoryImage(
                            sha256=row["sha256"],
                            media_type=row["media_type"],
                            data=row["data"],
                            caption=row["caption"],
                        )
                    )
        return found

    # ---------------------------------------------------------------------- 内部

    def _init_schema(self) -> None:
        self._conn.executescript(_DDL)
        self._migrate_lexical_zh()
        stored = self._meta("embedding_dim")
        if stored is None:
            self._set_meta("embedding_dim", str(self._dim))
        elif stored != str(self._dim):
            logger.warning(
                "向量维度变化，重建向量表（文本与图片索引需要重新写入）",
                extra={"stored_dim": stored, "configured_dim": self._dim},
            )
            self._conn.executescript(
                "DROP TABLE IF EXISTS vec_text; DROP TABLE IF EXISTS vec_image;"
            )
            self._set_meta("embedding_dim", str(self._dim))
        self._conn.executescript(_vector_ddl(self._dim))
        self._set_meta("schema_version", SCHEMA_VERSION)
        self._migrate_lexical_zh_version()

    def _migrate_lexical_zh_version(self) -> None:
        """分词口径变了（中文二元组 → 中英实词），重建中文 FTS 列。"""
        if self._meta("lexical_zh_version") == "2":
            return
        rows = self._conn.execute(
            "SELECT unit_rowid, id, user_id, lexical_full FROM memory_unit"
        ).fetchall()
        with self._conn:
            self._conn.execute("DELETE FROM fts_zh")
            for row in rows:
                lexical_zh = zh_index_text(str(row["lexical_full"]))
                self._conn.execute(
                    "UPDATE memory_unit SET lexical_zh = ? WHERE unit_rowid = ?",
                    (lexical_zh, row["unit_rowid"]),
                )
                self._conn.execute(
                    "INSERT INTO fts_zh (rowid, memory_id, user_id, lexical_zh)"
                    " VALUES (?, ?, ?, ?)",
                    (row["unit_rowid"], row["id"], row["user_id"], lexical_zh),
                )
        self._set_meta("lexical_zh_version", "2")
        logger.info("已按新口径重建中文词面索引", extra={"units": len(rows)})

    def _migrate_lexical_zh(self) -> None:
        """给老库补增量列与中文 FTS 索引（启动时自动执行，幂等）。"""
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(memory_unit)").fetchall()
        }
        added: list[str] = []
        for name, ddl in (
            ("lexical_zh", "TEXT NOT NULL DEFAULT ''"),
            ("lexical_full", "TEXT NOT NULL DEFAULT ''"),
            ("transcripts", "TEXT NOT NULL DEFAULT '[]'"),
        ):
            if name not in columns:
                self._conn.execute(f"ALTER TABLE memory_unit ADD COLUMN {name} {ddl}")
                added.append(name)
        caption_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(image_caption)").fetchall()
        }
        if "transcript" not in caption_columns:
            self._conn.execute(
                "ALTER TABLE image_caption ADD COLUMN transcript TEXT NOT NULL DEFAULT ''"
            )
            added.append("image_caption.transcript")
        if added:
            logger.info("数据库增量迁移", extra={"added": ",".join(added)})
        self._conn.execute(
            "UPDATE memory_unit SET lexical_full = lexical"
            " WHERE lexical_full = '' AND lexical != ''"
        )
        missing = self._conn.execute(
            "SELECT unit_rowid, id, user_id, lexical, lexical_zh FROM memory_unit"
        ).fetchall()
        indexed = {
            int(row["rowid"]) for row in self._conn.execute("SELECT rowid FROM fts_zh").fetchall()
        }
        pending = [
            (row["unit_rowid"], row["id"], row["user_id"], row["lexical"], row["lexical_zh"])
            for row in missing
            if int(row["unit_rowid"]) not in indexed
        ]
        if not pending:
            return
        with self._conn:
            for unit_rowid, unit_id, user_id, lexical, stored_zh in pending:
                lexical_zh = stored_zh or zh_index_text(lexical)
                if not stored_zh:
                    self._conn.execute(
                        "UPDATE memory_unit SET lexical_zh = ? WHERE unit_rowid = ?",
                        (lexical_zh, unit_rowid),
                    )
                self._conn.execute(
                    "INSERT INTO fts_zh (rowid, memory_id, user_id, lexical_zh)"
                    " VALUES (?, ?, ?, ?)",
                    (unit_rowid, unit_id, user_id, lexical_zh),
                )
        logger.info("已补齐中文词面索引", extra={"units": len(pending)})

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def _row_to_unit(row: sqlite3.Row) -> StoredUnit:
    try:
        captions = tuple(json.loads(row["captions"] or "[]"))
    except json.JSONDecodeError:
        captions = ()
    try:
        transcripts = tuple(json.loads(row["transcripts"] or "[]"))
    except json.JSONDecodeError:
        transcripts = ()
    return StoredUnit(
        unit_rowid=int(row["unit_rowid"]),
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        session_id=str(row["session_id"]),
        request_id=str(row["request_id"]),
        ordinal=int(row["ordinal"]),
        role=Role(row["role"]),
        ts_ms=None if row["ts_ms"] is None else int(row["ts_ms"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        text=str(row["text"]),
        captions=captions,
        transcripts=transcripts,
        lexical=str(row["lexical"]),
        lexical_full=str(row["lexical_full"]),
        lexical_zh=str(row["lexical_zh"]),
        image_count=int(row["image_count"]),
    )


def _chunks(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Cursor]:
    """显式事务：一次 Add 的所有行要么一起可见，要么一起回滚。"""
    conn.execute("BEGIN IMMEDIATE")
    cursor = conn.cursor()
    try:
        yield cursor
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
    finally:
        cursor.close()


_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_unit (
    unit_rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    ts_ms INTEGER,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    captions TEXT NOT NULL DEFAULT '[]',
    transcripts TEXT NOT NULL DEFAULT '[]',
    lexical TEXT NOT NULL,
    lexical_full TEXT NOT NULL DEFAULT '',
    lexical_zh TEXT NOT NULL DEFAULT '',
    image_count INTEGER NOT NULL DEFAULT 0,
    embedding_state TEXT NOT NULL DEFAULT 'ok',
    caption_state TEXT NOT NULL DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_unit_scope
    ON memory_unit (user_id, session_id, request_id);
CREATE INDEX IF NOT EXISTS idx_unit_user_time
    ON memory_unit (user_id, ts_ms);

CREATE TABLE IF NOT EXISTS memory_image (
    image_rowid INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memory_unit(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    data BLOB NOT NULL,
    UNIQUE (memory_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_image_memory ON memory_image (memory_id);
CREATE INDEX IF NOT EXISTS idx_image_sha ON memory_image (sha256);

CREATE TABLE IF NOT EXISTS image_caption (
    sha256 TEXT PRIMARY KEY,
    media_type TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    caption TEXT NOT NULL,
    transcript TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_word USING fts5(
    memory_id UNINDEXED,
    user_id UNINDEXED,
    lexical,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_trigram USING fts5(
    memory_id UNINDEXED,
    user_id UNINDEXED,
    lexical,
    tokenize = 'trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_zh USING fts5(
    memory_id UNINDEXED,
    user_id UNINDEXED,
    lexical_zh,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _vector_ddl(dim: int) -> str:
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS vec_text USING vec0(
    unit_rowid INTEGER PRIMARY KEY,
    user_id TEXT PARTITION KEY,
    embedding FLOAT[{dim}] distance_metric=cosine
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_image USING vec0(
    image_rowid INTEGER PRIMARY KEY,
    user_id TEXT PARTITION KEY,
    embedding FLOAT[{dim}] distance_metric=cosine
);
"""
