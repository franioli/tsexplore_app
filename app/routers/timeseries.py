import json
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..config import get_logger, get_settings
from ..services.inversion import invert_node2, load_dic_data
from ..services.plots import make_timeseries_figure
from ..services.provider import get_data_provider
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
):
    """Return timeseries figure for a node and optional inversion overlay."""
    logger.info(f"timeseries: node=({node_x},{node_y}) inversion={ts_inversion}")

    provider = get_data_provider()
    all_data = provider.preload_all()

    # Build time series for this node across all dates
    ts = _extract_node_timeseries(all_data, node_x, node_y, provider)

    if not ts or not ts.get("dates"):
        raise HTTPException(status_code=404, detail="No timeseries data for node")

    # core plotting inputs
    dates = [datetime.strptime(d, "%Y%m%d") for d in ts["dates"]]
    comp_list = [c.strip() for c in components.split(",")]
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

    metadata = {"Node": f"({node_x:.1f},{node_y:.1f})", "N dates": len(dates)}
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
            # Build KDTree and find node
            tree, coords = build_kdtree(provider)
            dist, idx = tree.query([node_x, node_y], k=1)
            if not np.isfinite(dist):
                raise RuntimeError("node not found in KDTree")

            # load full dic data and locate node
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
            logger.error(f"Inversion overlay error: {e}")

    return JSONResponse(json.loads(fig.to_json()))


def _extract_node_timeseries(
    all_data: dict[str, dict],
    node_x: float,
    node_y: float,
    provider,
) -> dict:
    """
    Extract time series for a specific node across all dates.

    Uses KDTree to find nearest node index, then extracts data for that index
    across all dates.
    """
    if not all_data:
        return {}

    # Build KDTree to find node
    tree, coords = build_kdtree(provider)
    dist, idx = tree.query([node_x, node_y], k=1)

    if not np.isfinite(dist):
        logger.warning(f"Node ({node_x}, {node_y}) not found in KDTree")
        return {}

    # Extract time series
    ts = {
        "dates": [],
        "dx": [],
        "dy": [],
        "disp_mag": [],
        "u": [],
        "v": [],
        "V": [],
        "ensamble_mad": [],
    }

    for date_str in sorted(all_data.keys()):
        data = all_data[date_str]
        ts["dates"].append(date_str)
        ts["dx"].append(data["dx"][idx])
        ts["dy"].append(data["dy"][idx])
        ts["disp_mag"].append(data["disp_mag"][idx])
        ts["u"].append(data["u"][idx])
        ts["v"].append(data["v"][idx])
        ts["V"].append(data["V"][idx])
        ts["ensamble_mad"].append(data["ensamble_mad"][idx])

    return ts
