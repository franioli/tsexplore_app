from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import get_logger
from ..services.data import cache

logger = get_logger()
router = APIRouter()


@router.get("/ping")
async def ping():
    return JSONResponse({"status": "ok"})


@router.get("/status")
async def status():
    return JSONResponse(
        {
            "status": "ready"
            if cache.is_loaded() and cache.has_kdtree()
            else "starting",
            "data_loaded": cache.is_loaded(),
            "kdtree_built": cache.has_kdtree(),
            "num_dates": cache.num_dates,
        }
    )
