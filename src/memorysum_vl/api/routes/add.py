from fastapi import APIRouter, Depends, status

from memorysum_vl.api.deps import AddServiceDep
from memorysum_vl.api.mappers import to_add_response, to_conversation
from memorysum_vl.api.security import require_api_token
from memorysum_vl.schemas.add import AddRequest, AddResponse
from memorysum_vl.schemas.errors import ErrorResponse

router = APIRouter(tags=["memories"], dependencies=[Depends(require_api_token)])


@router.post(
    "/add",
    response_model=AddResponse,
    status_code=status.HTTP_200_OK,
    responses={401: {"model": ErrorResponse}, 501: {"model": ErrorResponse}},
)
async def add_memories(payload: AddRequest, service: AddServiceDep) -> AddResponse:
    receipt = await service.add(to_conversation(payload))
    return to_add_response(receipt)
