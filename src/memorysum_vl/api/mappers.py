from collections.abc import Sequence

from memorysum_vl.core.config import Track
from memorysum_vl.core.errors import UnsupportedContentError
from memorysum_vl.core.media import InlineImage, build_inline_image_data_uri, parse_inline_image
from memorysum_vl.domain.models import (
    Content,
    Conversation,
    ImageContent,
    MemoryEntry,
    Message,
    Role,
    SearchQuery,
    TextContent,
    WriteReceipt,
)
from memorysum_vl.schemas.add import AddMessage, AddRequest, AddResponse
from memorysum_vl.schemas.content import (
    ContentPart,
    ImageContentPart,
    ImageUrl,
    MessageContent,
    TextContentPart,
)
from memorysum_vl.schemas.search import SearchHit, SearchRequest, SearchResponse

TEXT_JOINER = "\n"


def to_conversation(request: AddRequest) -> Conversation:
    return Conversation(
        request_id=request.request_id,
        user_id=request.user_id,
        session_id=request.session_id,
        messages=tuple(_to_message(message) for message in request.messages),
    )


def to_search_query(request: SearchRequest) -> SearchQuery:
    return SearchQuery(
        parts=_to_domain_parts(request.query),
        user_id=request.user_id,
        top_k=request.top_k,
        options=tuple(request.options or ()),
    )


def to_add_response(receipt: WriteReceipt) -> AddResponse:
    return AddResponse(
        success=True,
        request_id=receipt.request_id,
        user_id=receipt.user_id,
        session_id=receipt.session_id,
    )


def to_search_response(entries: Sequence[MemoryEntry], track: Track) -> SearchResponse:
    return SearchResponse(data=[_to_search_hit(entry, track) for entry in entries])


def _to_message(message: AddMessage) -> Message:
    return Message(
        role=Role(message.role),
        parts=_to_domain_parts(message.content),
        timestamp_ms=message.timestamp,
    )


def _to_domain_parts(content: MessageContent) -> tuple[Content, ...]:
    if isinstance(content, str):
        return (TextContent(text=content),)
    return tuple(_to_domain_part(part) for part in content)


def _to_domain_part(part: ContentPart) -> Content:
    if isinstance(part, TextContentPart):
        return TextContent(text=part.text)
    return _to_image_content(part)


def _to_image_content(part: ImageContentPart) -> ImageContent:
    image = parse_inline_image(part.image_url.url)
    return ImageContent(media_type=image.media_type, data=image.data)


def _to_search_hit(entry: MemoryEntry, track: Track) -> SearchHit:
    return SearchHit(
        id=entry.id,
        content=_to_api_content(entry.parts, track),
        score=entry.score,
        created_at=entry.created_at,
    )


def _to_api_content(parts: Sequence[Content], track: Track) -> MessageContent:
    if track is Track.MULTIMODAL:
        return [_to_content_part(part) for part in parts]
    return _join_text_parts(parts)


def _to_content_part(part: Content) -> ContentPart:
    match part:
        case TextContent(text=text):
            return TextContentPart(type="text", text=text)
        case ImageContent(media_type=media_type, data=data):
            image = InlineImage(media_type=media_type, data=data)
            return ImageContentPart(
                type="image_url",
                image_url=ImageUrl(url=build_inline_image_data_uri(image)),
            )


def _join_text_parts(parts: Sequence[Content]) -> str:
    texts: list[str] = []
    for part in parts:
        match part:
            case TextContent(text=text):
                texts.append(text)
            case ImageContent():
                raise UnsupportedContentError(
                    "multimodal memory requires the multimodal track to be selected"
                )
    return TEXT_JOINER.join(texts)
