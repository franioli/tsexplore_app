"""FastAPI application for velocity field visualization and time series analysis."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .data import (
    build_global_kdtree,
    format_date_for_display,
    get_loaded_dates,
    load_all_series,
    load_day_dic,
    nearest_node,
    preload_all_data,
)
from .models import HealthCheckResponse, NearestNodeRequest, NearestNodeResponse
from .plots import make_timeseries_figure, make_velocity_map_figure

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_ts_inversion(
    data_dir,
    file_pattern,
    filename_pattern,
    node_x,
    node_y,
):
    from .inversion import invert_node2, load_dic_data

    logger.info("Running time series inversion for single node...")
    # Load all DIC data once
    # TODO: DO NOT READ all the data every time, optimize this
    dic_data = load_dic_data(data_dir)

    # Get node index from KDTree
    tree, coords = build_global_kdtree(data_dir, file_pattern, filename_pattern)
    dist, node_idx = tree.query([node_x, node_y], k=1)
    if not np.isfinite(dist):
        raise ValueError(f"Could not locate node at ({node_x}, {node_y})")
    logger.debug(f"Found node index {node_idx} at distance {dist:.2f}px")

    # Extract data for this node
    ew_series = dic_data["ew"][:, node_idx]
    ns_series = dic_data["ns"][:, node_idx]
    ensamble_mad = dic_data["weight"][:, node_idx]
    timestamp = dic_data["timestamp"]

    # Run inversion for this node
    logger.info(
        f"Starting inversion computation at node index {node_idx} - ({node_x:.1f}, {node_y:.1f})"
    )
    inversion_results = invert_node2(
        ew_series=ew_series,
        ns_series=ns_series,
        timestamp=timestamp,
        node_idx=node_idx,
        node_x=coords[node_idx, 0],
        node_y=coords[node_idx, 1],
        weight_method="variable",
        weight_variable=ensamble_mad,
        regularization_method="laplacian",
        lambda_scaling=1.0,
        iterates=10,
    )

    if inversion_results is None:
        raise ValueError("Inversion returned None (invalid data)")

    # Unpack inversion results
    ew_hat = inversion_results["EW_hat"]
    ns_hat = inversion_results["NS_hat"]
    time_hat = inversion_results["Time_hat"]  # Shape: (n_inverted, 3)
    logger.info(
        "Inversion results obtained successfully: "
        f"   Inversion output: {len(ew_hat)} time steps, "
    )

    return ew_hat, ns_hat, time_hat


# Application state class for better state management
class AppState:
    """Centralized application state."""

    def __init__(self):
        self.data_loaded = False
        self.kdtree_built = False

    def is_ready(self) -> bool:
        """Check if app is ready to serve requests."""
        return self.data_loaded and self.kdtree_built


# Global app state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: preload data on startup."""
    logger.info("Starting application - preloading data...")
    try:
        preload_all_data(
            data_dir=settings.data_dir,
            file_pattern=settings.file_pattern,
            filename_pattern=settings.filename_pattern,
            dt_days_preferred=settings.dt_days_preferred,
        )
        app_state.data_loaded = True
        logger.info("Data preloaded successfully")

        build_global_kdtree(
            data_dir=settings.data_dir,
            file_pattern=settings.file_pattern,
            filename_pattern=settings.filename_pattern,
        )
        app_state.kdtree_built = True
        logger.info("KDTree built successfully")

    except Exception as e:
        logger.error(f"Failed to preload data: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down application")


# Initialize FastAPI app
app = FastAPI(
    title="Velocity Field Time Series Viewer",
    description="Interactive visualization of velocity fields and time series data",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ===== HTML Endpoints =====
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page."""
    if not app_state.is_ready():
        return HTMLResponse(
            content="<h1>Service starting up, please wait...</h1>", status_code=503
        )

    dates_raw = get_loaded_dates()  # YYYYMMDD format

    # Convert to display format (DD/MM/YYYY) for user
    dates_display = [
        datetime.strptime(d, "%Y%m%d").strftime("%d/%m/%Y") for d in dates_raw
    ]

    # Convert dates to iso format for HTML5 date inputs (YYYY-MM-DD)
    dates_display_iso = [
        datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates_raw
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "dates_raw": dates_raw,
            "dates_display": dates_display,
            "dates_iso": dates_display_iso,
            "background_image": settings.background_image,
        },
    )


@app.get("/api/dates")
async def api_dates():
    """Get list of available dates."""
    dates = get_loaded_dates()
    return JSONResponse({"dates": dates})


@app.get("/api/velocity-map")
async def api_velocity_map(
    date: str = Query(..., description="Date in YYYYMMDD format"),
    use_velocity: bool = Query(True),
    cmin: float | None = Query(None, ge=0),
    cmax: float | None = Query(None, ge=0),
    colorscale: str = Query(settings.default_colorscale),
    plot_type: str = Query(settings.default_plot_type),
    marker_size: int = Query(settings.default_marker_size, ge=1, le=20),
    marker_opacity: float = Query(settings.default_marker_opacity, ge=0, le=1),
    downsample_points: int = Query(settings.default_downsample_points, ge=100),
    selected_x: float | None = Query(None),
    selected_y: float | None = Query(None),
):
    """Velocity map endpoint."""
    logger.info(f"API /velocity-map - date={date}, use_velocity={use_velocity}")

    if plot_type not in {"scatter", "raster"}:
        logger.warning(f"Invalid plot_type '{plot_type}', defaulting to 'scatter'")
        plot_type = "scatter"

    try:
        # Load raw data (has both displacement and velocity precomputed)
        raw_data = load_day_dic(
            settings.data_dir,
            date,
            settings.file_pattern,
            settings.filename_pattern,
        )

        # Select which data to plot based on flag
        x = raw_data["x"]
        y = raw_data["y"]
        dt_days = raw_data.get("dt_days", None)
        # ensable_nmad = raw_data.get("nmad", None)  # Not used currently

        if use_velocity:
            logger.debug("Selected velocity data for plotting")
            mag = raw_data["V"]
            # u_comp = raw_data["u"]  # Not used currently
            # v_comp = raw_data["v"]  # Not used currently
            units = "velocity (px/day)"
        else:
            logger.debug("Selected displacement data for plotting")
            mag = raw_data["disp_mag"]
            # u_comp = raw_data["dx"]  # Not used currently
            # v_comp = raw_data["dy"]  # Not used currently
            units = "displacement (px)"

        # Prepare selected node if provided
        selected_node = None
        if selected_x is not None and selected_y is not None:
            selected_node = {"x": selected_x, "y": selected_y}

        # Build metadata dictionary
        metadata = {
            "Date": format_date_for_display(date),
            "dt (days)": dt_days if dt_days else "N/A",
            "N points": len(x),
            "Median |v|": float(np.median(mag)),
            "MAD |v|": float(np.median(np.abs(mag - np.median(mag)))),
            "Min |v|": float(np.min(mag)),
            "Max |v|": float(np.max(mag)),
        }
        # Create the velocity map figure
        fig = make_velocity_map_figure(
            x=x,
            y=y,
            mag=mag,
            std=None,
            plot_type=plot_type,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            downsample_points=downsample_points,
            selected_node=selected_node,
            units=units,
            background_image_path=settings.background_image,
            metadata=metadata,
        )

        logger.info("velocity map plot created successfully")
        return JSONResponse(fig.to_dict())

    except FileNotFoundError as e:
        logger.error(f"Date {date} not found: {e}")
        return JSONResponse({"error": f"No data for date {date}"}, status_code=404)
    except Exception as e:
        logger.error(f"Error creating velocity map plot: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/timeseries")
async def api_timeseries(
    node_x: float,
    node_y: float,
    use_velocity: bool = True,
    components: str = "V",
    marker_mode: str = "lines+markers",
    xmin_date: str | None = None,
    xmax_date: str | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    show_error_band: bool = False,
    ts_inversion: bool = False,
):
    """Time series endpoint."""
    logger.info(
        f"API /timeseries - node=({node_x}, {node_y}), use_velocity={use_velocity}"
    )

    try:
        # Load time series for the node
        ts = load_all_series(
            settings.data_dir,
            settings.file_pattern,
            settings.filename_pattern,
            node_x=node_x,
            node_y=node_y,
        )

        if not ts["dates"]:
            logger.warning("No data found for time series")
            raise HTTPException(status_code=404, detail="No data found for this node")

        # Convert dates to datetime objects
        dates = [datetime.strptime(d, "%Y%m%d") for d in ts["dates"]]

        # Select which data to plot
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

        # Build metadata
        metadata = {
            "Node": f"({node_x:.1f}, {node_y:.1f})",
            "N dates": len(dates),
        }
        if "V" in comp_list:
            metadata["Mean |v|"] = float(np.mean(V))
            metadata["Std |v|"] = float(np.std(V))

        # Handle error bands
        u_std = None
        v_std = None
        V_std = None
        if show_error_band and ts.get("ensamble_mad") is not None:
            V_std = np.array(ts.get("ensamble_mad"))
            metadata["Error metric"] = "Ensamble MAD"
            metadata["Average Ensamble MAD"] = float(np.mean(ts["ensamble_mad"]))

        # Prepare node coordinates
        node_coords = {"x": node_x, "y": node_y}

        # Create figure
        fig = make_timeseries_figure(
            dates=dates,
            u=u,
            v=v,
            V=V,
            u_std=u_std,
            v_std=v_std,
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

        # Add inversion trace if requested
        if ts_inversion:
            logger.info("Running time series inversion...")
            try:
                from plotly import graph_objects as go

                from .inversion import invert_node2, load_dic_data

                # Load DIC data and run inversion
                dic_data = load_dic_data(settings.data_dir)
                tree, coords = build_global_kdtree(
                    settings.data_dir,
                    settings.file_pattern,
                    settings.filename_pattern,
                )
                dist, node_idx = tree.query([node_x, node_y], k=1)
                if not np.isfinite(dist):
                    raise ValueError(f"Could not locate node at ({node_x}, {node_y})")

                # Extract node data
                ew_series = dic_data["ew"][:, node_idx]
                ns_series = dic_data["ns"][:, node_idx]
                ensamble_mad = dic_data["weight"][:, node_idx]
                timestamp = dic_data["timestamp"]

                # Run inversion
                inv_results = invert_node2(
                    ew_series=ew_series,
                    ns_series=ns_series,
                    timestamp=timestamp,
                    node_idx=node_idx,
                    node_x=coords[node_idx, 0],
                    node_y=coords[node_idx, 1],
                    weight_method="variable",
                    weight_variable=ensamble_mad,
                    regularization_method="laplacian",
                    lambda_scaling=1.0,
                    iterates=10,
                )

                if inv_results:
                    ew_hat = inv_results["EW_hat"]
                    ns_hat = inv_results["NS_hat"]
                    time_hat = inv_results["Time_hat"]

                    dates_inv = [
                        np.datetime_as_string(t, unit="D") for t in time_hat[:, 1]
                    ]
                    V_inv = np.sqrt(ew_hat**2 + ns_hat**2)

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
                logger.error(f"Inversion failed: {e}", exc_info=True)
                # Continue without inversion trace

        logger.info("Time series plot created successfully")

        return JSONResponse(json.loads(fig.to_json()))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating time series: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/nearest", response_model=NearestNodeResponse)
async def api_nearest(
    date: str = Query(...),
    x: float = Query(...),
    y: float = Query(...),
    radius: float = Query(10.0, ge=0, le=1000),
    method: str = Query("hybrid"),
):
    """Find nearest node."""
    # Validate request
    request_data = NearestNodeRequest(date=date, x=x, y=y, radius=radius, method=method)

    logger.info(
        f"API /nearest - date={date}, x={x}, y={y}, radius={radius}, method={method}"
    )
    try:
        node = nearest_node(
            x=x,
            y=y,
            date=date,
            data_dir=settings.data_dir,
            file_pattern=settings.file_pattern,
            filename_pattern=settings.filename_pattern,
            radius=radius,
            method=method,
        )

        if node is None:
            logger.warning(f"No node found within radius {radius}")
            raise HTTPException(status_code=404, detail="No node within radius")

        logger.info(f"Found nearest node: ({node['x']:.2f}, {node['y']:.2f})")
        return NearestNodeResponse(x=float(node["x"]), y=float(node["y"]))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding nearest node: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    from .data import cache

    return HealthCheckResponse(
        status="healthy" if app_state.is_ready() else "starting",
        data_loaded=cache.is_loaded(),
        kdtree_built=cache.has_kdtree(),
        num_dates=cache.num_dates,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
