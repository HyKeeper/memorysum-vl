from fastapi import APIRouter

from memorysum_vl.api.routes import add, search


def build_memory_router(prefix: str) -> APIRouter:
    router = APIRouter(prefix=prefix)
    router.include_router(add.router)
    router.include_router(search.router)
    return router
