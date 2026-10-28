import base64
import binascii
from dataclasses import dataclass

from memorysum_vl.core.limits import MAX_IMAGE_BYTES, SUPPORTED_IMAGE_MEDIA_TYPES

DATA_URI_SCHEME = "data:"
BASE64_SEPARATOR = ";base64,"


class MediaFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InlineImage:
    media_type: str
    data: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class ImageDataUri:
    media_type: str
    payload: str

    @property
    def decoded_size(self) -> int:
        padding = len(self.payload) - len(self.payload.rstrip("="))
        return (len(self.payload) // 4) * 3 - padding


def inspect_inline_image(url: str) -> ImageDataUri:
    if not url.startswith(DATA_URI_SCHEME):
        raise MediaFormatError("image url must be an inline base64 data uri")
    header, separator, payload = url.partition(BASE64_SEPARATOR)
    if not separator:
        raise MediaFormatError("image data uri must be base64 encoded")
    media_type = header.removeprefix(DATA_URI_SCHEME).lower()
    if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_MEDIA_TYPES))
        raise MediaFormatError(f"image media type {media_type!r} is not one of: {supported}")
    uri = ImageDataUri(media_type=media_type, payload=payload)
    if uri.decoded_size == 0:
        raise MediaFormatError("image payload must not be empty")
    if uri.decoded_size > MAX_IMAGE_BYTES:
        raise MediaFormatError(f"image size {uri.decoded_size} exceeds {MAX_IMAGE_BYTES} bytes")
    return uri


def parse_inline_image(url: str) -> InlineImage:
    uri = inspect_inline_image(url)
    try:
        data = base64.b64decode(uri.payload, validate=True)
    except binascii.Error as exc:
        raise MediaFormatError("image payload is not valid base64") from exc
    return InlineImage(media_type=uri.media_type, data=data)


def build_inline_image_data_uri(image: InlineImage) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"{DATA_URI_SCHEME}{image.media_type}{BASE64_SEPARATOR}{encoded}"
