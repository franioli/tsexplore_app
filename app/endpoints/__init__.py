from fastapi import APIRouter

from . import health as health_router
from . import inversion as inversion_router
from . import loader as loader_router
from . import nearest as nearest_router
from . import timeseries as timeseries_router
from . import velocitymap as map_router

router = APIRouter()
router.include_router(health_router.router)
router.include_router(loader_router.router)
router.include_router(map_router.router)
router.include_router(nearest_router.router)
router.include_router(timeseries_router.router)
router.include_router(inversion_router.router)
