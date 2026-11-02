"""图片预处理：算指纹、按模型上限缩放、转 data URI。

原始字节完整入库（回看原图、审计都用得上），送去模型的是缩过的副本：
Qwen3-VL 按 32 px 切 patch，长边 1280 大约 1024 个视觉 token，正好落在
embedding server 单次 ubatch 的预算内，也避免大图把 caption 拖到几十秒。
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from typing import Final

from memorysum_vl.core.media import DATA_URI_SCHEME, InlineImage

try:  # Pillow 缺失时退回原图，功能不受影响，只是慢一些
    from PIL import Image, ImageOps

    PILLOW_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover - 依赖已声明，兜底路径只做保险
    PILLOW_AVAILABLE = False

DEFAULT_MAX_SIDE: Final[int] = 1280
DEFAULT_JPEG_QUALITY: Final[int] = 90
PASSTHROUGH_LIMIT: Final[int] = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PreparedImage:
    media_type: str
    data: bytes
    width: int
    height: int
    resized: bool


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_for_model(
    image: InlineImage,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> PreparedImage:
    """把图片压到长边 `max_side` 以内，统一成 JPEG。"""
    if not PILLOW_AVAILABLE:
        return PreparedImage(image.media_type, image.data, 0, 0, False)
    try:
        with Image.open(io.BytesIO(image.data)) as handle:
            handle.load()
            width, height = handle.size
            if (
                image.media_type == "image/jpeg"
                and max(width, height) <= max_side
                and image.size_bytes <= PASSTHROUGH_LIMIT
            ):
                return PreparedImage(image.media_type, image.data, width, height, False)
            frame = ImageOps.exif_transpose(handle).convert("RGB")
            resized = max(frame.size) > max_side
            if resized:
                frame.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=quality, optimize=True)
            return PreparedImage("image/jpeg", buffer.getvalue(), *frame.size, resized)
    except Exception:  # noqa: BLE001 - 解码失败就退回原图，让下游模型自己报错
        return PreparedImage(image.media_type, image.data, 0, 0, False)


def to_base64(image: InlineImage) -> str:
    return base64.b64encode(image.data).decode("ascii")


def to_data_uri(media_type: str, data: bytes) -> str:
    return f"{DATA_URI_SCHEME}{media_type};base64,{base64.b64encode(data).decode('ascii')}"
