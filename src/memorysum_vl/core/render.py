"""把一条消息渲染成「可检索文本」与「可返回内容」。

两个产物刻意分开：

* 检索文本（lexical）带时间与角色，喂向量、BM25、重排三路，让「上周三谁说的」
  这类时间/说话人线索有落点；
* 返回内容（content）默认也带同样的前缀，但去掉前缀的开关留着——比赛方只要求
  `content` 非空且按名次返回，前缀是不是噪声最终由评测结果说话。
"""

from __future__ import annotations

from datetime import UTC, datetime

from memorysum_vl.domain.models import Role

IMAGE_LABEL = "[image]"
TRANSCRIPT_LABEL = "[image text]"
ASSISTANT_LABEL = "assistant"
USER_LABEL = "user"


def format_timestamp(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    try:
        moment = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return moment.strftime("%Y-%m-%d %H:%M")


def role_label(role: Role) -> str:
    return USER_LABEL if role is Role.USER else ASSISTANT_LABEL


def render_text(
    *,
    role: Role,
    timestamp_ms: int | None,
    text: str,
    captions: tuple[str, ...] = (),
    transcripts: tuple[str, ...] = (),
    include_timestamp: bool = True,
    include_role: bool = True,
    include_captions: bool = True,
    include_transcripts: bool = True,
    session_label: str | None = None,
) -> str:
    """一行元信息 + 正文 + 每张图的描述与逐条转写。"""
    lines: list[str] = []
    body = text.strip()
    if body:
        lines.append(body)
    if include_captions:
        for index, caption in enumerate(captions):
            cleaned = caption.strip()
            lines.append(f"{IMAGE_LABEL} {cleaned}" if cleaned else IMAGE_LABEL)
            if include_transcripts and index < len(transcripts):
                transcribed = transcripts[index].strip()
                if transcribed:
                    lines.append(f"{TRANSCRIPT_LABEL} {transcribed}")
    elif not body:
        lines.append(IMAGE_LABEL)

    prefix_parts: list[str] = []
    stamp = format_timestamp(timestamp_ms) if include_timestamp else None
    if session_label:
        prefix_parts.append(session_label)
    elif stamp:
        prefix_parts.append(f"[{stamp} UTC]")
    if include_role:
        prefix_parts.append(f"{role_label(role)}:")
    prefix = " ".join(prefix_parts)
    joined = "\n".join(lines) or IMAGE_LABEL
    return f"{prefix} {joined}" if prefix else joined
