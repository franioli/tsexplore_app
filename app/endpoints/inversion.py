import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, Response

from ..cache import cache
from ..config import get_logger, get_settings
from ..models import SaveInversionRequest, TraceRequest
from ..services.data_provider import get_data_provider
from ..services.inversion import invert_node

logger = get_logger()
settings = get_settings()

router = APIRouter()


def _compute_charrier_weights(
    provider,
    cache,
    node_x: float,
    node_y: float,
    ew: np.ndarray,
    ns: np.ndarray,
    final_dates: np.ndarray,  # datetime64[D]
    timestamp: np.ndarray,
    k: int = 9,
    min_neighbors: int = 4,
    eps: float = 1e-6,
    dt_days: int | None = None,
    dt_hours_tolerance: float = 0.0,
) -> np.ndarray | None:
    """
    Return per-observation Charrier weights (length = n_obs) or None on failure.
    Requires neighbor time series that match `final_dates` (simple, robust).
    """
    # KDTree and coords must exist in cache
    if cache.kdtree is None:
        return None
    tree, coords = cache.kdtree
    dists, idxs = tree.query([node_x, node_y], k=min(k, coords.shape[0]))
    if np.isscalar(idxs):
        idxs = [int(idxs)]
    neighbors = []
    for idx in np.atleast_1d(idxs):
        cx, cy = coords[int(idx)]
        ts = provider.extract_node_timeseries(
            cx, cy, dt_days=dt_days, dt_hours_tolerance=dt_hours_tolerance
        )
        if not ts:
            continue
        neigh_dates = np.asarray(ts["reference_dates"], dtype="datetime64[D]")
        # Simple requirement: neighbour must have same reference dates as target
        if neigh_dates.shape == final_dates.shape and np.all(
            neigh_dates == final_dates
        ):
            neighbors.append(ts)
    if len(neighbors) < min_neighbors:
        return None

    # Build matrices (n_neighbors x n_obs)
    n_obs = final_dates.shape[0]
    Iew = np.vstack([np.asarray(n["dx"], dtype=np.float32) for n in neighbors])
    Ins = np.vstack([np.asarray(n["dy"], dtype=np.float32) for n in neighbors])

    # deltat in days (float)
    deltat = np.abs(
        (timestamp[:, 1] - timestamp[:, 0]).astype("timedelta64[D]").astype(float)
    )
    # avoid divide by zero
    deltat = np.where(deltat == 0, 1.0, deltat).astype(np.float32)

    rates_ew = Iew / deltat[np.newaxis, :]
    rates_ns = Ins / deltat[np.newaxis, :]

    tew = np.nanmedian(rates_ew)
    tns = np.nanmedian(rates_ns)

    denom_ew = np.median(np.abs(ew - tew)) + eps
    denom_ns = np.median(np.abs(ns - tns)) + eps
    Zew = np.abs(ew - tew) / denom_ew
    Zns = np.abs(ns - tns) / denom_ns

    mad = 1.0 / (Zew * Zns + 0.004)
    mad = mad / (np.nanmax(mad) + eps)

    median_complex = np.median(rates_ew + 1j * rates_ns, axis=0)
    node_complex = ew.astype(np.complex64) + 1j * ns.astype(np.complex64)
    denom = (np.abs(node_complex) * np.abs(median_complex)) + eps
    MA = ((node_complex * np.conj(median_complex)) / denom).real
    MA[MA < 0] = 0.0

    weights = mad * MA
    s = np.sum(weights)
    if s <= 0:
        return None
    return (weights / s).astype(np.float32)


def _pick_unused_color(
    used_colors: list[str] | None, palette: list[str] | None = None
) -> str:
    """Pick a color from palette that is not present in used_colors.
    If none available, generate a random hex color.
    """
    import random

    if palette is None:
        # a sensible palette (Plotly default-ish)
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    used_set = {c.strip().lower() for c in (used_colors or []) if isinstance(c, str)}
    candidates = [c for c in palette if c.lower() not in used_set]
    if candidates:
        return random.choice(candidates)
    # fallback: random hex
    return f"#{random.getrandbits(24):06x}"


@router.post(
    "/inversion/run",
    summary="Run time-series inversion (node series provided)",
    tags=["inversion"],
)
async def run_node_inversion(
    node_x: float = Body(..., description="Node X coordinate"),
    node_y: float = Body(..., description="Node Y coordinate"),
    weight_method: str | None = Body(..., description="Weighting method"),
    regularization_method: str = Body("laplacian", description="Regularization method"),
    lambda_scaling: float = Body(..., description="Scaling for lambda"),
    iterates: int = Body(10, ge=1, description="Number of inversion iterations"),
    date_min: str | None = Body(
        None, description="Optional minimum date filter (YYYY-MM-DD)"
    ),
    date_max: str | None = Body(
        None,
        description="Optional maximum date filter (YYYY-MM-DD)",
    ),
):
    """
    Run time-series inversion on provided time-series data.
    """
    # Fail early if no data loaded
    if not cache.is_loaded():
        logger.warning("Nearest requested but no data has been loaded")
        raise HTTPException(
            status_code=400,
            detail="No data loaded yet. Press 'Load' before requesting nearest node.",
        )

    logger.info(f"Performing inversion for node at ({node_x}, {node_y})")
    logger.debug(f"Inversion request date_min={date_min} date_max={date_max}")

    try:
        logger.info("Fetching time series data")
        provider = get_data_provider()

        # Do not group by dt: we need the full list of observations for inversion
        ts_data = provider.extract_node_timeseries(
            node_x=node_x, node_y=node_y, dt_days=None
        )
        if not ts_data:
            raise ValueError("No time-series data available for the requested node")

        # Unpack data arrays
        final_dates = np.asarray(ts_data["final_dates"], dtype="datetime64[D]")
        initial_dates = np.asarray(ts_data["initial_dates"], dtype="datetime64[D]")

        # Build EW/NS series (displacements) and timestamp array (n_obs, 2)
        dx = np.asarray(ts_data["dx"], dtype=np.float32)
        dy = np.asarray(ts_data["dy"], dtype=np.float32)

        # extract the timeseries within optional date range
        if date_min or date_max:
            logger.info(
                f"Applying date filter: min={date_min or 'none'}, max={date_max or 'none'}"
            )
            mask = np.ones(final_dates.shape[0], dtype=bool)
            if date_min:
                date_min_dt = np.datetime64(date_min, "D")
                mask &= final_dates >= date_min_dt
            if date_max:
                date_max_dt = np.datetime64(date_max, "D")
                mask &= final_dates <= date_max_dt

            # Apply mask to all arrays
            dx = dx[mask]
            dy = dy[mask]
            final_dates = final_dates[mask]
            initial_dates = initial_dates[mask]

        n_obs = dx.shape[0]
        if n_obs < 1:
            raise ValueError("Not enough observations for inversion")

        if final_dates.shape[0] != n_obs or initial_dates.shape[0] != n_obs:
            raise ValueError("Inconsistent date arrays in timeseries data")

        timestamp = np.empty((n_obs, 2), dtype="datetime64[D]")
        timestamp[:, 0] = initial_dates
        timestamp[:, 1] = final_dates

        if weight_method == "variable":
            # Per-observation ensemble MAD weights ot use if weight_method is set to 'variable'
            ens_mad = ts_data.get("ensemble_mad", None)
            if ens_mad is None or len(ens_mad) != n_obs:
                logger.warning(
                    "Ensemble MAD array is missing or inconsistent in timeseries data. Fallback to 'residuals' for initial weights"
                )
                weight_var = None
                weight_method = "residuals"
            else:
                weight_var = np.asarray(ens_mad, dtype=np.float32)
        elif weight_method == "charrier":
            logger.info("Computing Charrier weights...")
            weight_var = _compute_charrier_weights(
                provider=provider,
                cache=cache,
                node_x=node_x,
                node_y=node_y,
                ew=dx,
                ns=dy,
                final_dates=final_dates,
                timestamp=timestamp,
                k=9,
                min_neighbors=4,
                eps=1e-6,
                dt_days=None,
                dt_hours_tolerance=0.0,
            )
            if weight_var is None:
                logger.warning(
                    "Failed to compute Charrier weights. Fallback to 'residuals' for initial weights"
                )
                weight_method = "residuals"
            else:
                logger.info(
                    "Charrier weights computed. Passing them as weight variable to inversion."
                )
                weight_method = "variable"
        else:
            weight_var = None

        logger.info("Data prepared (%d observations). Running inversion...", n_obs)
        res = invert_node(
            ew_series=dx,
            ns_series=dy,
            timestamp=timestamp,
            weight_method=weight_method,
            weight_variable=weight_var,
            regularization_method=regularization_method,
            lambda_scaling=lambda_scaling,
            iterates=iterates,
        )
        if not res:
            raise RuntimeError("invert_node returned no result")

        # normalize numpy arrays to plain JSON-friendly lists
        try:
            ew_hat = np.asarray(res["EW_hat"]).tolist()
            ns_hat = np.asarray(res["NS_hat"]).tolist()
            # Time_hat may be numpy datetime; extract day-string per sample
            time_hat = np.asarray(res["Time_hat"])
            try:
                dates_arr = time_hat[:, 1]
            except Exception:
                dates_arr = time_hat
            dates = [str(d) for d in np.datetime_as_string(dates_arr, unit="D")]
            V_hat = (np.sqrt(np.array(ew_hat) ** 2 + np.array(ns_hat) ** 2)).tolist()
        except Exception:
            ew_hat = res.get("EW_hat")
            ns_hat = res.get("NS_hat")
            dates = res.get("Time_hat")
            V_hat = None

        logger.info("Inversion completed.")

        return JSONResponse(
            {
                "status": "ok",
                "node_inversion": {
                    "EW_hat": ew_hat,
                    "NS_hat": ns_hat,
                    "dates": dates,
                    "V_hat": V_hat,
                },
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        logger.warning("Inversion not supported: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        logger.exception("node inversion failed")
        raise HTTPException(status_code=500, detail="inversion failure")


@router.post(
    "/inversion/trace", summary="Build a plotly trace from node inversion output"
)
async def build_inversion_trace(req: TraceRequest = Body(...)):
    """Return a Plotly trace JSON for the provided inversion output."""
    inv = req.node_inversion
    if not inv.dates:
        raise HTTPException(status_code=400, detail="Missing dates in node_inversion")

    # compute V_hat if missing but EW_hat/NS_hat present
    if inv.V_hat is None:
        if inv.EW_hat is not None and inv.NS_hat is not None:
            V = (
                np.sqrt(np.array(inv.EW_hat) ** 2 + np.array(inv.NS_hat) ** 2)
            ).tolist()
        else:
            raise HTTPException(
                status_code=400,
                detail="V_hat missing and EW_hat/NS_hat not available to compute it",
            )
    else:
        V = inv.V_hat

    if req.overlay_color:
        color = req.overlay_color
    else:
        # prefer a color not already used in the plot (frontend may send used_colors)
        color = _pick_unused_color(req.used_colors)
        # keep the previous convention for refresh_plot when palette is exhausted or not provided
        if not req.used_colors and not req.overlay_color:
            color = "darkred" if req.refresh_plot else "orange"

    trace = {
        "x": inv.dates,
        "y": V,
        "mode": req.mode,
        "name": req.name or "|v| (Inverted)",
        "line": {"color": color, "width": 2},
        "marker": {"size": 6, "symbol": req.marker_symbol},
    }
    return JSONResponse({"trace": trace})


@router.post("/inversion/save", summary="Save node inversion to a txt/csv file")
async def save_inversion(req: SaveInversionRequest = Body(...)):
    """Return a text file (attachment) containing the inversion results.

    Format: CSV with header:
      date,x,y,EW_hat,NS_hat,V_hat
    One row per sample in node_inversion.dates. Missing components are left empty.
    """
    inv = req.node_inversion
    dates = inv.dates or []
    n = len(dates)

    # basic consistency checks
    if inv.EW_hat is not None and len(inv.EW_hat) != n:
        raise HTTPException(status_code=400, detail="EW_hat length mismatch")
    if inv.NS_hat is not None and len(inv.NS_hat) != n:
        raise HTTPException(status_code=400, detail="NS_hat length mismatch")
    if inv.V_hat is not None and len(inv.V_hat) != n:
        raise HTTPException(status_code=400, detail="V_hat length mismatch")

    lines = ["date,x,y,EW_hat,NS_hat,V_hat"]
    for i, d in enumerate(dates):
        ew = "" if inv.EW_hat is None else f"{float(inv.EW_hat[i]):.6g}"
        ns = "" if inv.NS_hat is None else f"{float(inv.NS_hat[i]):.6g}"
        if inv.V_hat is not None:
            v = f"{float(inv.V_hat[i]):.6g}"
        elif inv.EW_hat is not None and inv.NS_hat is not None:
            v = f"{np.hypot(inv.EW_hat[i], inv.NS_hat[i]):.6g}"
        else:
            v = ""
        lines.append(f"{d},{req.node_x},{req.node_y},{ew},{ns},{v}")

    content = "\n".join(lines) + "\n"
    filename = (
        req.filename or f"node_inversion_x{int(req.node_x)}_y{int(req.node_y)}.txt"
    )
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "text/plain; charset=utf-8",
    }
    return Response(content=content.encode("utf-8"), headers=headers)
