from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from memorysum_vl.schemas.content import MessageContent, NonEmptyText, enforce_content_budget


class AddMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: MessageContent
    timestamp: int | None = None

    @model_validator(mode="after")
    def _enforce_content_budget(self) -> Self:
        enforce_content_budget(self.content)
        return self


class AddRequest(BaseModel):
    request_id: NonEmptyText
    messages: list[AddMessage] = Field(min_length=1)
    user_id: NonEmptyText
    session_id: NonEmptyText


class AddResponse(BaseModel):
    success: Literal[True]
    request_id: str
    user_id: str
    session_id: str
