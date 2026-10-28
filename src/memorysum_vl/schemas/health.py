from typing import Literal

from pydantic import BaseModel

from memorysum_vl.core.config import Track


class HealthResponse(BaseModel):
    status: Literal["ok"]
    track: Track
