from typing import Final

MAX_IMAGE_BYTES: Final[int] = 10 * 1024 * 1024
MAX_REQUEST_IMAGE_BYTES: Final[int] = 30 * 1024 * 1024
MAX_RETURNED_MEMORIES: Final[int] = 100
MIN_TOP_K: Final[int] = 1
SUPPORTED_IMAGE_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
