from fastapi import APIRouter, Depends, status

from memorysum_vl.api.deps import SearchServiceDep, SettingsDep
from memorysum_vl.api.mappers import to_search_query, to_search_response
from memorysum_vl.api.security import require_api_token
from memorysum_vl.schemas.errors import ErrorResponse
from memorysum_vl.schemas.search import SearchRequest, SearchResponse

router = APIRouter(tags=["memories"], dependencies=[Depends(require_api_token)])


@router.post(
    "/search",
    response_model=SearchResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    responses={401: {"model": ErrorResponse}, 501: {"model": ErrorResponse}},
)
async def search_memories(
    payload: SearchRequest,
    service: SearchServiceDep,
    settings: SettingsDep,
) -> SearchResponse:
    entries = await service.search(to_search_query(payload))
    return to_search_response(entries, settings.track)
