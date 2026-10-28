from fastapi import APIRouter

from memorysum_vl.api.deps import SettingsDep
from memorysum_vl.schemas.health import HealthResponse

router = APIRouter(tags=["operations"])


@router.get("/healthz", response_model=HealthResponse)
async def read_health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", track=settings.track)
