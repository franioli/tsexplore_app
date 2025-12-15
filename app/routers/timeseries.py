import json
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..config import get_logger, get_settings
from ..services.data_provider import DataProvider, get_data_provider
from ..services.inversion import invert_node2, load_dic_data
from ..services.plots import make_timeseries_figure
from ..services.spatial import build_kdtree

logger = get_logger()
settings = get_settings()

router = APIRouter()


@router.get("/")
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
    ts_inversion: bool = Query(False),
    dt_days: int | None = Query(None, ge=0, description="Select record by dt_days"),
    prefer_dt_days: int | None = Query(None, ge=0, description="Pick closest dt_days"),
    prefer_dt_tolerance: int | None = Query(
        None, ge=0, description="Tolerance (days) for closest dt"
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
    logger.info("timeseries: node=(%s,%s) inversion=%s", node_x, node_y, ts_inversion)

    provider = get_data_provider()
    dates_available = provider.get_available_dates()

    ts = _extract_node_timeseries(
        dates=dates_available,
        node_x=node_x,
        node_y=node_y,
        provider=provider,
        dt_days=dt_days,
        prefer_dt_days=prefer_dt_days,
        prefer_dt_tolerance=prefer_dt_tolerance,
    )

    if not ts or not ts.get("dates"):
        raise HTTPException(status_code=404, detail="No timeseries data for node")

    # core plotting inputs
    dates = [datetime.strptime(d, "%Y%m%d") for d in ts["dates"]]
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

    V_std = (
        np.array(ts.get("ensamble_mad"))
        if show_error_band and ts.get("ensamble_mad")
        else None
    )

    metadata: dict[str, Any] = {
        "Node": f"({node_x:.1f},{node_y:.1f})",
        "N dates": len(dates),
    }
    if dt_days is not None:
        metadata["dt (days)"] = int(dt_days)
    elif prefer_dt_days is not None:
        metadata["prefer dt (days)"] = int(prefer_dt_days)
    if master_date is not None:
        metadata["master"] = master_date

    if "V" in comp_list:
        metadata["Mean |v|"] = float(np.mean(V))
        metadata["Std |v|"] = float(np.std(V))

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
        node_coords={"x": node_x, "y": node_y},
        y_label=y_label,
        xmin_date=xmin_date,
        xmax_date=xmax_date,
        ymin=ymin,
        ymax=ymax,
        metadata=metadata,
    )

    # add simple inversion overlay (magnitude only)
    if ts_inversion:
        logger.warning(
            "Performing time-series inversion works only on local data. Working on db is not implemented yet."
        )
        try:
            tree, coords = build_kdtree(provider)
            dist, idx = tree.query([node_x, node_y], k=1)
            if not np.isfinite(dist):
                raise RuntimeError("node not found in KDTree")

            try:
                dic = load_dic_data(settings.data_dir)
            except FileNotFoundError as e:
                raise RuntimeError(f"load_dic_data error: {e}")

            ew_series = dic["ew"][:, idx]
            ns_series = dic["ns"][:, idx]
            ens_mad = dic["weight"][:, idx] if "weight" in dic else None
            timestamp = dic["timestamp"]

            inv = invert_node2(
                ew_series=ew_series,
                ns_series=ns_series,
                timestamp=timestamp,
                node_idx=idx,
                node_x=coords[idx, 0],
                node_y=coords[idx, 1],
                weight_method="variable" if ens_mad is not None else "residuals",
                weight_variable=ens_mad,
                regularization_method="laplacian",
                lambda_scaling=1.0,
                iterates=10,
            )

            if inv:
                ew_hat = inv["EW_hat"]
                ns_hat = inv["NS_hat"]
                time_hat = inv["Time_hat"]
                dates_inv = [np.datetime_as_string(t, unit="D") for t in time_hat[:, 1]]
                V_inv = np.sqrt(np.array(ew_hat) ** 2 + np.array(ns_hat) ** 2)

                from plotly import graph_objects as go

                fig.add_trace(
                    go.Scattergl(
                        x=dates_inv,
                        y=V_inv,
                        mode="lines+markers",
                        name="|v| (Inverted)",
                        line=dict(color="darkred", width=2),
                        marker=dict(size=6, symbol="diamond"),
                    )
                )

        except Exception as e:
            logger.error("Inversion overlay error: %s", e)

    return JSONResponse(json.loads(fig.to_json()))


def _extract_node_timeseries(
    *,
    dates: list[str],
    node_x: float,
    node_y: float,
    provider: DataProvider,
    dt_days: int | None = None,
    prefer_dt_days: int | None = None,
    prefer_dt_tolerance: int | None = None,
) -> dict[str, list[Any]]:
    """Extract a node time series across available slave dates.

    Uses a KDTree to find the nearest node index, then for each slave date
    fetches the selected DIC record via `provider.get_dic_data(...)` and extracts
    the values at that index.

    Args:
        dates: Available slave dates as `YYYYMMDD`.
        node_x: Query x coordinate.
        node_y: Query y coordinate.
        provider: Active data provider (file or database).
        dt_days: Optional exact dt (days) selection per slave date.
        prefer_dt_days: Optional preferred dt (days) selection per slave date.
        prefer_dt_tolerance: Optional tolerance (days) for closest dt.

    Returns:
        A dict of lists with keys:
          `dates, dx, dy, disp_mag, u, v, V, ensamble_mad`.
        Returns an empty dict if no data is available.
    """
    if not dates:
        return {}

    # Build KDTree to find nearest node index (coordinate set is shared across records)
    tree, _coords = build_kdtree(provider)
    dist, idx = tree.query([node_x, node_y], k=1)

    if not np.isfinite(dist):
        logger.warning("Node (%s, %s) not found in KDTree", node_x, node_y)
        return {}

    ts: dict[str, list[Any]] = {
        "dates": [],
        "dx": [],
        "dy": [],
        "disp_mag": [],
        "u": [],
        "v": [],
        "V": [],
        "ensamble_mad": [],
    }

    for date_str in sorted(dates):
        data = provider.get_dic_data(
            date_str,
            dt_days=dt_days,
            prefer_dt_days=prefer_dt_days,
            prefer_dt_tolerance=prefer_dt_tolerance,
        )
        if not data:
            continue

        ts["dates"].append(date_str)
        ts["dx"].append(float(data["dx"][idx]))
        ts["dy"].append(float(data["dy"][idx]))
        ts["disp_mag"].append(float(data["disp_mag"][idx]))
        ts["u"].append(float(data["u"][idx]))
        ts["v"].append(float(data["v"][idx]))
        ts["V"].append(float(data["V"][idx]))

        em = data.get("ensamble_mad")
        ts["ensamble_mad"].append(float(em[idx]) if em is not None else 0.0)

    return ts
