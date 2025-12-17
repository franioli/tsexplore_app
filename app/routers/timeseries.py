import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..cache import cache
from ..config import get_logger, get_settings
from ..services.data_provider import get_data_provider
from ..services.plots import add_trace_with_error_band, make_timeseries_figure

logger = get_logger()
settings = get_settings()

router = APIRouter()

DELTA_DAYS = [1, 3, 5, 10]  # TODO: this is hardcoded here now, move to config


@router.get("/timeseries", summary="Get timeseries figure", tags=["timeseries"])
async def timeseries(
    node_x: float = Query(...),
    node_y: float = Query(...),
    use_velocity: bool = Query(True),
    components: str = Query("V"),
    marker_mode: str = Query("lines+markers"),
    xmin_date: str | None = Query(None),
    xmax_date: str | None = Query(None),
    ymin: float | None = Query(None),
    ymax: float | None = Query(None),
    show_error_band: bool = Query(False),
    dt_days: int | None = Query(None, ge=0, description="Select record by dt_days"),
    group_by_dt: bool = Query(
        True, description="Group time series by dt (True = group)"
    ),
    delta_days: list[int] | None = Query(
        DELTA_DAYS,
        description="Optional list of dt group centers, e.g. ?delta_days=1&delta_days=3",
    ),
):
    """Return timeseries figure for a node and optional inversion overlay.

    Args:
        node_x: Query x coordinate.
        node_y: Query y coordinate.
        use_velocity: If True, plot velocity components; otherwise plot displacement.
        components: Comma-separated list of components to plot (e.g. "u,v,V").
        marker_mode: Plotly scatter mode (e.g. "lines+markers").
        xmin_date: Optional minimum date bound (passed to plotting).
        xmax_date: Optional maximum date bound (passed to plotting).
        ymin: Optional y-axis minimum.
        ymax: Optional y-axis maximum.
        show_error_band: If True, add an error band if available.
        ts_inversion: If True, attempt inversion overlay (local data only).
        dt_days: If provided, select only records with this dt (days) for each slave date.
        prefer_dt_days: If provided, pick the closest dt (days) per slave date.
        prefer_dt_tolerance: Optional tolerance (days) for closest dt.

    Returns:
        A Plotly figure JSON response.

    Raises:
        HTTPException: If no timeseries data is available for the requested node.
    """
    logger.info(f"timeseries: node=({node_x},{node_y})")

    # Fail early if no data loaded
    if not cache.is_loaded():
        logger.warning("Nearest requested but no data has been loaded")
        raise HTTPException(
            status_code=400,
            detail="No data loaded yet. Press 'Load' before requesting nearest node.",
        )

    provider = get_data_provider()

    ts_groups = provider.extract_node_timeseries(
        node_x=node_x,
        node_y=node_y,
        dt_days=dt_days,
        group_by_dt=group_by_dt,
        delta_days=delta_days,
    )
    if not ts_groups:
        raise HTTPException(status_code=404, detail="No timeseries data for node")

    # If only a single group is present, use that as the main ts, otherwise prepare to overlay groups
    group_keys = sorted(ts_groups.keys())
    main_key = group_keys[0]
    ts = ts_groups[main_key]

    # Get dates
    dates = ts.get("reference_dates")
    if dates is None:
        raise HTTPException(
            status_code=500,
            detail="Timeseries data missing 'reference_dates' field",
        )
    dates = [str(d) for d in np.datetime_as_string(dates, unit="D")]

    # core plotting inputs
    comp_list = [c.strip() for c in components.split(",") if c.strip()]

    if use_velocity:
        u = np.array(ts["u"])
        v = np.array(ts["v"])
        V = np.array(ts["V"])
        y_label = "Velocity (px/day)"
    else:
        u = np.array(ts["dx"])
        v = np.array(ts["dy"])
        V = np.array(ts["disp_mag"])
        y_label = "Displacement (px)"

    V_std = None
    if show_error_band:
        ensemble = ts.get("ensemble_mad", None)
        if ensemble is not None:
            arr = np.asarray(ensemble)
            if arr.size > 0:
                V_std = arr

    # Build metadata info
    metadata: dict[str, Any] = {
        "Node": f"({node_x:.1f},{node_y:.1f})",
        "N dates": len(dates),
    }
    if len(group_keys) > 1:
        metadata["dt_days_groups"] = [int(k) for k in group_keys]

    if "V" in comp_list:
        metadata["Mean |v|"] = float(np.mean(V))
        metadata["Std |v|"] = float(np.std(V))

    title = f"Time series - Node: ({node_x:.1f}, {node_y:.1f})"

    fig = make_timeseries_figure(
        dates=dates,
        u=u,
        v=v,
        V=V,
        u_std=None,
        v_std=None,
        V_std=V_std,
        components=comp_list,
        marker_mode=marker_mode,
        y_label=y_label,
        xmin_date=xmin_date,
        xmax_date=xmax_date,
        ymin=ymin,
        ymax=ymax,
        metadata=metadata,
        title=title,
    )

    # If multiple dt groups present, overlay them (reuse add_trace_with_error_band)
    if len(group_keys) > 1:
        palette = ["orange", "green", "purple", "cyan", "magenta", "yellow"]
        for i, k in enumerate(group_keys[1:], start=0):
            g = ts_groups[k]
            dates_i = [
                str(d) for d in np.datetime_as_string(g["reference_dates"], unit="D")
            ]

            # choose color per group (cycle palette)
            color = palette[i % len(palette)]

            if use_velocity:
                u_i = np.asarray(g["u"])
                v_i = np.asarray(g["v"])
                V_i = np.asarray(g["V"])
            else:
                u_i = np.asarray(g["dx"])
                v_i = np.asarray(g["dy"])
                V_i = np.asarray(g["disp_mag"])

            ens_i = None
            if show_error_band:
                ens = g.get("ensemble_mad", None)
                if ens is not None:
                    arr = np.asarray(ens)
                    if arr.size > 0:
                        ens_i = arr

            # Add traces for requested components
            if "u" in comp_list:
                add_trace_with_error_band(
                    fig=fig,
                    x=dates_i,
                    y=u_i,
                    y_std=None,
                    name=f"u (dt={k}d)",
                    color=color,
                    marker_mode=marker_mode,
                )
            if "v" in comp_list:
                add_trace_with_error_band(
                    fig=fig,
                    x=dates_i,
                    y=v_i,
                    y_std=None,
                    name=f"v (dt={k}d)",
                    color=color,
                    marker_mode=marker_mode,
                )
            if "V" in comp_list:
                add_trace_with_error_band(
                    fig=fig,
                    x=dates_i,
                    y=V_i,
                    y_std=ens_i,
                    name=f"|v| (dt={k}d)",
                    color=color,
                    marker_mode=marker_mode,
                )

    return JSONResponse(json.loads(fig.to_json()))
