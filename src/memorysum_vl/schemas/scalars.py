from datetime import UTC, datetime
from typing import Annotated

from pydantic import AwareDatetime, PlainSerializer


def _serialize_utc_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


UtcTimestamp = Annotated[
    AwareDatetime,
    PlainSerializer(_serialize_utc_timestamp, return_type=str, when_used="json"),
]
