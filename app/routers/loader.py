from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from ..cache import cache
from ..config import get_logger, get_settings
from ..services import get_data_provider

logger = get_logger()
settings = get_settings()

router = APIRouter()


@router.post("/load-range")
async def load_range(
    start_date: str,
    end_date: str,
    background_tasks: BackgroundTasks,
):
    """Start preloading a date range (DB backend only). Returns 202 and starts background task."""
    if not settings.use_database:
        raise HTTPException(
            status_code=400, detail="Load range is only available with DB backend."
        )

    provider = get_data_provider()

    # quick validation of dates
    try:
        # will raise ValueError if invalid
        from datetime import datetime

        for s in (start_date, end_date):
            if "-" in s:
                datetime.strptime(s, "%Y-%m-%d")
            else:
                datetime.strptime(s, "%Y%m%d")
    except Exception:
        raise HTTPException(
            status_code=400, detail="Invalid date format; use YYYYMMDD or YYYY-MM-DD"
        )

    def progress_callback(done, total):
        cache.load_done = done
        cache.load_total = total

    def _do_load():
        try:
            provider.preload_range(
                start_date, end_date, progress_callback=progress_callback
            )
        except Exception as e:
            logger.error("DB preload failed", exc_info=True)
            cache.load_in_progress = False
            cache.load_error = str(e)

    # ensure previous load not in progress
    if cache.load_in_progress:
        raise HTTPException(
            status_code=409, detail="Another loading operation is in progress."
        )

    cache.load_in_progress = True
    cache.load_total = 0
    cache.load_done = 0
    cache.load_error = None
    cache.load_start_date = start_date
    cache.load_end_date = end_date

    background_tasks.add_task(_do_load)
    return JSONResponse(
        {"status": "accepted", "start_date": start_date, "end_date": end_date},
        status_code=202,
    )


@router.get("/progress")
async def progress() -> JSONResponse:
    """Return progress metadata for the current load task."""
    return JSONResponse(
        {
            "in_progress": cache.load_in_progress,
            "total": cache.load_total,
            "done": cache.load_done,
            "start_date": cache.load_start_date,
            "end_date": cache.load_end_date,
            "error": cache.load_error,
        }
    )
