import threading
import traceback

from fastapi import APIRouter, HTTPException

from ..config import get_logger, get_settings
from ..services import DataProvider, get_data_provider

logger = get_logger()
settings = get_settings()

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
        # If no dates provided -> load all (both backends should support load_all)
        if not start_date and not end_date:
            provider.load_all()
        else:
            # Accept missing bounds too (interpret as open interval)
            provider.load_range(
                start_date or "", end_date or "", progress_callback=progress
            )
    except Exception as e:
        # Log full traceback and store it in the state so the frontend can show details
        tb = traceback.format_exc()
        logger.error(f"Background load failed: {e}\n{tb}")
        with _lock:
            _state["error"] = tb
            _state["in_progress"] = False

    finally:
        with _lock:
            _state["in_progress"] = False


@router.post("/loader/load", tags=["loader"])
def load_data(start_date: str | None = None, end_date: str | None = None):
    logger.debug(f"API /loader/load called start_date={start_date} end_date={end_date}")

    with _lock:
        if _state["in_progress"]:
            raise HTTPException(status_code=409, detail="Load already in progress")
        _state.update({"in_progress": True, "done": 0, "total": 0, "error": None})

    # Validate provider early to fail fast if misconfigured
    try:
        provider = get_data_provider()
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
