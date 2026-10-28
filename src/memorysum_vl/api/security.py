from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from memorysum_vl.api.deps import ContainerDep
from memorysum_vl.core.errors import UnauthorizedError

bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")

WWW_AUTHENTICATE = {"WWW-Authenticate": "Bearer"}


async def require_api_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    container: ContainerDep,
) -> None:
    if credentials is None or not container.api_token.matches(credentials.credentials):
        raise UnauthorizedError("missing or invalid bearer token", headers=WWW_AUTHENTICATE)
