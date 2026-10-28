from typing import Self

from pydantic import BaseModel, Field, model_validator

from memorysum_vl.core.limits import MAX_RETURNED_MEMORIES, MIN_TOP_K
from memorysum_vl.schemas.content import MessageContent, NonEmptyText, enforce_content_budget
from memorysum_vl.schemas.scalars import UtcTimestamp


class SearchRequest(BaseModel):
    query: MessageContent
    options: list[NonEmptyText] | None = None
    user_id: NonEmptyText
    top_k: int = Field(ge=MIN_TOP_K, le=MAX_RETURNED_MEMORIES)

    @model_validator(mode="after")
    def _enforce_content_budget(self) -> Self:
        enforce_content_budget(self.query)
        return self


class SearchHit(BaseModel):
    id: NonEmptyText
    content: MessageContent
    score: float | None = None
    created_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _enforce_content_budget(self) -> Self:
        enforce_content_budget(self.content)
        return self


class SearchResponse(BaseModel):
    data: list[SearchHit]
