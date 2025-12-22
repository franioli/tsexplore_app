import threading
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import Settings, get_logger
from ..services import DataProvider, get_data_provider

logger = get_logger()

router = APIRouter()

_lock = threading.Lock()
_state = {"in_progress": False, "done": 0, "total": 0, "error": None}


def worker(
    provider: DataProvider, start_date: str | None, end_date: str | None
) -> None:
    def progress(done: int, total: int):
        with _lock:
            _state["done"] = int(done)
            _state["total"] = int(total)

    try:
        # If no dates provided -> load all
        if not start_date and not end_date:
            provider.load_all()
        else:
            # Accept missing bounds too (interpret as open interval)
            provider.load_range(
                start_date or "", end_date or "", progress_callback=progress
            )
    except Exception as e:
        # Log full traceback and store it in the state
        tb = traceback.format_exc()
        logger.error(f"Background load failed: {e}\n{tb}")
        with _lock:
            _state["error"] = tb
            _state["in_progress"] = False

    finally:
        with _lock:
            _state["in_progress"] = False


@router.post("/loader/load", tags=["loader"])
def load_data(
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: str | None = None,
    file_search_pattern: str | None = None,
    filename_date_template: str | None = None,
    background_image: str | None = None,
    dt_days: str | None = None,
    dt_hours_tolerance: str | None = None,
    invert_y: bool | None = None,
) -> dict[str, Any]:
    logger.debug(
        f"""API /loader/load called start_date={start_date} end_date={end_date} 
        data_dir={data_dir} file_search_pattern={file_search_pattern} 
        filename_date_template={filename_date_template} dt_days={dt_days}
        dt_hours_tolerance={dt_hours_tolerance}
        background_image={background_image}
        """,
    )

    # Apply runtime non-persistent overrides to settings
    settings = Settings()
    try:
        if data_dir:
            settings.data_dir = Path(data_dir)
        if file_search_pattern:
            settings.file_search_pattern = file_search_pattern
        if filename_date_template:
            settings.filename_date_template = filename_date_template
        if background_image is not None:
            settings.background_image = background_image or None
        if dt_days:
            # allow "1,3,5" or "3"
            parts = [p.strip() for p in dt_days.split(",") if p.strip() != ""]
            if len(parts) == 1:
                settings.dt_days = int(parts[0])
            else:
                settings.dt_days = [int(p) for p in parts]
        if dt_hours_tolerance:
            settings.dt_hours_tolerance = int(dt_hours_tolerance)
        if invert_y is not None:
            settings.invert_y = invert_y
        logger.debug(f"Applied runtime settings overrides: {settings}")
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid parameter value: {e}"
        ) from e

    # Start loading if not already in progress
    with _lock:
        if _state["in_progress"]:
            raise HTTPException(status_code=409, detail="Load already in progress")
        _state.update({"in_progress": True, "done": 0, "total": 0, "error": None})

    # Validate provider early to fail fast if misconfigured
    try:
        provider = get_data_provider(settings=settings or None)
        _ = provider.get_available_dates()
    except Exception as e:
        logger.exception("Provider initialization failed")
        raise HTTPException(status_code=500, detail=f"Provider init error: {e}") from e

    # Start background thread
    threading.Thread(
        target=worker, args=(provider, start_date, end_date), daemon=True
    ).start()

    return {"status": "started"}


@router.get("/loader/progress", tags=["loader"])
def get_progress():
    with _lock:
        return dict(_state)
