from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, StringConstraints

from memorysum_vl.core.limits import MAX_REQUEST_IMAGE_BYTES
from memorysum_vl.core.media import inspect_inline_image, parse_inline_image

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


def _assert_inline_image(url: str) -> str:
    parse_inline_image(url)
    return url


InlineImageUrl = Annotated[NonEmptyText, AfterValidator(_assert_inline_image)]


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: NonEmptyText


class ImageUrl(BaseModel):
    url: InlineImageUrl


class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[TextContentPart | ImageContentPart, Field(discriminator="type")]

ContentPartList = Annotated[list[ContentPart], Field(min_length=1)]

MessageContent = NonEmptyText | ContentPartList


def content_image_bytes(content: MessageContent) -> int:
    if isinstance(content, str):
        return 0
    return sum(
        inspect_inline_image(part.image_url.url).decoded_size
        for part in content
        if isinstance(part, ImageContentPart)
    )


def enforce_content_budget(content: MessageContent) -> None:
    total = content_image_bytes(content)
    if total > MAX_REQUEST_IMAGE_BYTES:
        raise ValueError(f"images total {total} exceeds {MAX_REQUEST_IMAGE_BYTES} bytes")
