from fastapi import APIRouter

from . import health as health_router
from . import inversion as inversion_router
from . import nearest as nearest_router
from . import timeseries as timeseries_router
from . import velocitymap as map_router

router = APIRouter()
router.include_router(health_router.router, prefix="/health", tags=["health"])
router.include_router(map_router.router, prefix="/velocity-map", tags=["map"])
router.include_router(nearest_router.router, prefix="/nearest", tags=["nearest"])
router.include_router(
    timeseries_router.router, prefix="/timeseries", tags=["timeseries"]
)
router.include_router(inversion_router.router, prefix="/inversion", tags=["inversion"])
