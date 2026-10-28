from typing import Annotated

from fastapi import Depends, Request

from memorysum_vl.core.config import Settings
from memorysum_vl.core.container import Container
from memorysum_vl.services.add_service import AddService
from memorysum_vl.services.search_service import SearchService


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_app_settings(container: Annotated[Container, Depends(get_container)]) -> Settings:
    return container.settings


def get_add_service(container: Annotated[Container, Depends(get_container)]) -> AddService:
    return container.add_service


def get_search_service(container: Annotated[Container, Depends(get_container)]) -> SearchService:
    return container.search_service


ContainerDep = Annotated[Container, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
AddServiceDep = Annotated[AddService, Depends(get_add_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
