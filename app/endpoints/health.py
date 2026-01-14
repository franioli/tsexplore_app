from fastapi import APIRouter
from fastapi import status as http_status
from fastapi.responses import JSONResponse

from ..cache import cache
from ..config import get_logger
from ..dataloaders.data_provider import get_data_provider as _get_provider

logger = get_logger()
router = APIRouter()


@router.get(
    "/health",
    summary="Service health summary",
    tags=["health"],
)
async def status():
    """
    Lightweight health summary for dashboards/UI (does not perform expensive checks).
    Returns 200 even if some checks are degraded; use /health/ready for readiness.
    """
    return JSONResponse(
        {
            "data_loaded": cache.is_loaded(),
            "kdtree_built": cache.has_kdtree(),
            "num_records": cache.num_records,
        }
    )


@router.get("/health/ping")
async def ping():
    return JSONResponse({"status": "ok"})


@router.get("/health/fullcheck", summary="Full health check", tags=["health"])
async def fullcheck():
    """
    Health check endpoint: performs a set of lightweight checks and returns 200 if all
    checks pass, otherwise 503 with per-check details.

    Checks performed (in order):
      1) provider.get_available_dates() - must not raise; empty list is reported
      2) provider.load_range(...) for a single sample date - executed against a
         temporary cache so global state is not modified by this probe
      3) simple plotting-related checks:
         - velocity map data shape / presence check
         - timeseries extraction for a sample node
    """
    checks: dict = {}

    # Check if the data are loaded (global state)
    checks["cache_loaded"] = {
        "ok": bool(cache.is_loaded()),
        "num_records": cache.num_records,
    }
    checks["kdtree_built"] = {"ok": bool(cache.has_kdtree())}

    # 1) Probe provider.available_dates
    provider_ok = True
    provider_err = None
    sample_ok = None
    sample_err = None
    plot_velocity_ok = None
    plot_velocity_err = None
    plot_timeseries_ok = None
    plot_timeseries_err = None

    try:
        provider = _get_provider()
        dates = provider.get_available_dates()
        checks["available_dates"] = {"count": len(dates), "sample": dates[:3]}
        if not dates:
            provider_ok = False
            provider_err = "no available dates"
        else:
            # pick a sample date (YYYY-MM-DD as returned by provider)
            sample_date = dates[0]
            try:
                # Clear global cache and run sample load
                cache.clear()
                loaded = provider.load_range(
                    start_date=sample_date, end_date=sample_date
                )

                # Some providers return an int (loaded count), others use global cache;
                # treat both: prefer explicit returned value, otherwise check cache
                loaded_count = None
                if isinstance(loaded, int):
                    loaded_count = loaded
                else:
                    loaded_count = cache.num_records

                if loaded_count and loaded_count > 0:
                    sample_ok = True
                else:
                    sample_ok = False
                    sample_err = f"no data loaded for sample date {sample_iso}"

                # 3) Simple plotting checks on the sample data:
                # velocity map (presence of x/y and mag)
                try:
                    # try provider.get_dic_data for the sampled day (YYYYMMDD)
                    raw = provider.get_dic_data(sample_yyyymmdd)
                    if not raw:
                        raise RuntimeError("provider.get_dic_data returned empty")
                    # basic shape checks - avoid numpy truthiness
                    x = raw.get("x")
                    y = raw.get("y")
                    mag = (
                        raw.get("V")
                        if raw.get("V") is not None
                        else raw.get("disp_mag")
                    )

                    def _empty(seq):
                        if seq is None:
                            return True
                        try:
                            return len(seq) == 0
                        except Exception:
                            return True

                    if _empty(x) or _empty(y) or _empty(mag):
                        raise RuntimeError("insufficient data for velocity map")

                    # ensure consistent lengths
                    if len(x) != len(y) or len(x) != len(mag):
                        raise RuntimeError("inconsistent data lengths for velocity map")

                    plot_velocity_ok = True
                except Exception as e:
                    plot_velocity_ok = False
                    plot_velocity_err = str(e)

                # timeseries check: attempt to extract a timeseries for the first node
                try:
                    # import locally to avoid circular imports at module load
                    from .timeseries import _extract_node_timeseries

                    # use first point as sample node
                    node_x = float(x[0])
                    node_y = float(y[0])
                    ts = _extract_node_timeseries(
                        dates=[sample_yyyymmdd],
                        node_x=node_x,
                        node_y=node_y,
                        provider=provider,
                    )
                    if not ts or not ts.get("dates"):
                        raise RuntimeError("timeseries extraction returned empty")
                    plot_timeseries_ok = True
                except Exception as e:
                    plot_timeseries_ok = False
                    plot_timeseries_err = str(e)

            finally:
                # cleanup temporary load
                cache.clear()

    except Exception as exc:  # pragma: no cover - defensive catch
        provider_ok = False
        provider_err = str(exc)
        logger.exception("Data provider fullcheck probe failed")

    checks["data_provider"] = {
        "available_dates_ok": provider_ok,
        "available_dates_error": provider_err,
        "sample_load_ok": sample_ok,
        "sample_load_error": sample_err,
        "plot_velocity_ok": plot_velocity_ok,
        "plot_velocity_error": plot_velocity_err,
        "plot_timeseries_ok": plot_timeseries_ok,
        "plot_timeseries_error": plot_timeseries_err,
    }

    # overall: require provider available dates + sample load + plotting checks
    overall_ok = (
        provider_ok
        and sample_ok is True
        and plot_velocity_ok is True
        and plot_timeseries_ok is True
    )

    return JSONResponse(
        {"overall_ok": overall_ok, "checks": checks},
        status_code=http_status.HTTP_200_OK
        if overall_ok
        else http_status.HTTP_503_SERVICE_UNAVAILABLE,
    )
