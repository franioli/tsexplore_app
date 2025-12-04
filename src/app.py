import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .data import (
    build_global_kdtree,
    format_date_for_display,
    get_data_dir,
    get_file_pattern,
    get_filename_pattern,
    get_loaded_dates,
    load_all_series,
    load_day_dic,
    nearest_node,
    preload_all_data,
)
from .plots import make_timeseries_figure, make_velocity_map_figure

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Type aliases
ColorscalesType = Literal["Reds", "Viridis", "Plasma", "Blues", "RdYlBu", "Turbo"]
PlotTypesType = Literal["scatter", "raster"]
SearchMethodsType = Literal["hybrid", "kdtree", "grid"]

# Default values for API parameters
DEFAULT_USE_VELOCITY = True
DEFAULT_COLORSCALE = "Reds"
DEFAULT_PLOT_TYPE = "scatter"
DEFAULT_MARKER_SIZE = 6
DEFAULT_MARKER_OPACITY = 0.7
DEFAULT_DOWNSAMPLE_POINTS = 5000
DEFAULT_MARKER_MODE = "lines+markers"
DEFAULT_COMPONENTS = "V"
DEFAULT_SEARCH_METHOD = "hybrid"
DEFAULT_RADIUS = 10.0

# Load configuration from environment
BACKGROUND_IMAGE = os.getenv("BACKGROUND_IMAGE", "")
DT_DAYS_PREFERRED = int(os.getenv("DT_DAYS_PREFERRED", "3"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload all data on startup."""
    logger.info("Starting application...")
    data_dir = get_data_dir()
    file_pattern = get_file_pattern()
    filename_pattern = get_filename_pattern()

    # Log configuration
    if BACKGROUND_IMAGE:
        logger.info(f"Background image configured: {BACKGROUND_IMAGE}")
    else:
        logger.info("No background image configured")

    dt_days_preferred = DT_DAYS_PREFERRED
    logger.info(f"Preferred dt_days for velocity calculation: {dt_days_preferred}")
    try:
        preload_all_data(
            data_dir,
            file_pattern,
            filename_pattern,
            dt_days_preferred=dt_days_preferred,
        )
        build_global_kdtree(data_dir, file_pattern, filename_pattern)
        logger.info("Data preloading complete!")
    except Exception as e:
        logger.error(f"Failed to preload data: {e}", exc_info=True)

    yield  # App runs here

    logger.info("Shutting down application...")


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Rendering index page")

    try:
        # Get the actually loaded dates from cache
        dates_raw = get_loaded_dates()
        if not dates_raw:
            logger.warning("No dates loaded in cache")
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "dates_raw": [],
                    "dates_display": [],
                    "dates_iso": [],
                },
            )

        # Convert dates to display format and create mapping
        dates_display = []
        dates_iso = []

        for d in dates_raw:
            dates_display.append(format_date_for_display(d))
            dt = datetime.strptime(d, "%Y%m%d")
            dates_iso.append(dt.strftime("%Y-%m-%d"))

        logger.info(f"Found {len(dates_raw)} loaded dates")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "dates_raw": dates_raw,
                "dates_display": dates_display,
                "dates_iso": dates_iso,
            },
        )
    except Exception as e:
        logger.error(f"Error loading dates: {e}", exc_info=True)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "dates_raw": [],
                "dates_display": [],
                "dates_iso": [],
            },
        )


@app.get("/api/dates")
async def api_dates():
    """Return available dates in multiple formats (only actually loaded dates)."""
    dates_raw = get_loaded_dates()

    dates_info = []
    for d in dates_raw:
        dates_info.append(
            {
                "raw": d,
                "display": format_date_for_display(d),
                "iso": datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d"),
            }
        )

    return JSONResponse(dates_info)


@app.get("/api/velocity-map")
async def api_velocity_map(
    date: str,
    use_velocity: bool = DEFAULT_USE_VELOCITY,
    cmin: float | None = None,
    cmax: float | None = None,
    colorscale: ColorscalesType = DEFAULT_COLORSCALE,
    plot_type: PlotTypesType = DEFAULT_PLOT_TYPE,
    marker_size: int = DEFAULT_MARKER_SIZE,
    marker_opacity: float = DEFAULT_MARKER_OPACITY,
    downsample_points: int = DEFAULT_DOWNSAMPLE_POINTS,
    selected_x: float | None = None,
    selected_y: float | None = None,
    background_image_path: str = BACKGROUND_IMAGE,
):
    logger.info(f"API /velocity map - date={date}, use_velocity={use_velocity}")
    if plot_type not in {"scatter", "raster"}:
        logger.warning(f"Invalid plot_type '{plot_type}', defaulting to 'scatter'")
        plot_type = "scatter"

    data_dir = get_data_dir()
    file_pattern = get_file_pattern()
    filename_pattern = get_filename_pattern()

    try:
        # Load raw data (has both displacement and velocity precomputed)
        raw_data = load_day_dic(data_dir, date, file_pattern, filename_pattern)

        # Select which data to plot based on flag
        x = raw_data["x"]
        y = raw_data["y"]
        nmad = raw_data.get("nmad", None)

        if use_velocity:
            logger.debug("Selected velocity data for plotting")
            mag = raw_data["V"]
            u_comp = raw_data["u"]
            v_comp = raw_data["v"]
            units = "velocity (px/day)"
        else:
            logger.debug("Selected displacement data for plotting")
            mag = raw_data["disp_mag"]
            u_comp = raw_data["dx"]
            v_comp = raw_data["dy"]
            units = "displacement (px)"

        # Prepare selected node if provided
        selected_node = None
        if selected_x is not None and selected_y is not None:
            selected_node = {"x": selected_x, "y": selected_y}

        # Create the velocity map figure
        fig = make_velocity_map_figure(
            x=x,
            y=y,
            mag=mag,
            std=nmad,
            plot_type=plot_type,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            downsample_points=downsample_points,
            selected_node=selected_node,
            units=units,
            background_image_path=background_image_path,
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
    use_velocity: bool = DEFAULT_USE_VELOCITY,
    components: str = DEFAULT_COMPONENTS,
    marker_mode: str = DEFAULT_MARKER_MODE,
    xmin_date: str | None = None,
    xmax_date: str | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
):
    logger.info(
        f"API /timeseries - node=({node_x}, {node_y}), use_velocity={use_velocity}"
    )
    data_dir = get_data_dir()
    file_pattern = get_file_pattern()
    filename_pattern = get_filename_pattern()

    try:
        # Load time series for the node
        ts_raw = load_all_series(
            data_dir, file_pattern, filename_pattern, node_x=node_x, node_y=node_y
        )

        if not ts_raw["dates"]:
            logger.warning("No data found for time series")
            return JSONResponse(
                {"error": "No data found for this node"}, status_code=404
            )

        if not ts_raw["dates"]:
            logger.warning("No data found for time series")
            return JSONResponse(
                {"error": "No data found for this node"}, status_code=404
            )

        # Select which data to plot based on flag
        dates = ts_raw["dates"]
        comp_list = [c.strip() for c in components.split(",")]

        if use_velocity:
            logger.debug("Selected velocity data for time series")
            dt_days_list = ts_raw.get("dt_days", None)
            if dt_days_list is None:
                logger.error("dt_days information missing for velocity calculation")
                return JSONResponse(
                    {"error": "Velocity data not available"}, status_code=500
                )

            # Convert displacement to velocity
            u = [
                ui / dt if dt > 0 else ui
                for ui, dt in zip(ts_raw["u"], dt_days_list, strict=False)
            ]
            v = [
                vi / dt if dt > 0 else vi
                for vi, dt in zip(ts_raw["v"], dt_days_list, strict=False)
            ]
            V = [
                Vi / dt if dt > 0 else Vi
                for Vi, dt in zip(ts_raw["V"], dt_days_list, strict=False)
            ]
            y_label = "Velocity (px/day)"
        else:
            logger.debug("Selected displacement data for time series")
            u = ts_raw["u"]
            v = ts_raw["v"]
            V = ts_raw["V"]
            y_label = "Displacement (px)"

        node_coords = {"x": node_x, "y": node_y}

        # Create the time series figure
        fig = make_timeseries_figure(
            dates=dates,
            u=np.array(u),
            v=np.array(v),
            V=np.array(V),
            components=comp_list,
            marker_mode=marker_mode,
            node_coords=node_coords,
            y_label=y_label,
            xmin_date=xmin_date,
            xmax_date=xmax_date,
            ymin=ymin,
            ymax=ymax,
        )
        logger.info("Time series plot created successfully")
        return JSONResponse(fig.to_dict())

    except Exception as e:
        logger.error(f"Error creating time series: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/nearest")
async def api_nearest(
    date: str,
    x: float,
    y: float,
    radius: float = DEFAULT_RADIUS,
    method: SearchMethodsType = DEFAULT_SEARCH_METHOD,
):
    logger.info(
        f"API /nearest - date={date}, x={x}, y={y}, radius={radius}, method={method}"
    )
    data_dir = get_data_dir()
    file_pattern = get_file_pattern()
    filename_pattern = get_filename_pattern()

    try:
        node = nearest_node(
            data_dir,
            file_pattern,
            filename_pattern,
            date,
            x,
            y,
            radius=radius,
            method=method,  # Pass the method explicitly
        )

        if node is None:
            logger.warning(f"No node found within radius {radius}")
            return JSONResponse({"error": "No node within radius"}, status_code=404)

        logger.info(f"Found nearest node: ({node['x']:.2f}, {node['y']:.2f})")
        return JSONResponse({"x": float(node["x"]), "y": float(node["y"])})

    except Exception as e:
        logger.error(f"Error finding nearest node: {e}", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.post("/api/smooth")
async def api_smooth():
    logger.info("API /smooth called (not implemented)")
    return JSONResponse({"status": "not_implemented"})


# health check endpoint
@app.get("/api/health")
async def health_check():
    """Check if data is loaded and ready."""
    from .data import _ALL_DATA_CACHE, _KDTREE_CACHE

    status = {
        "status": "healthy",
        "data_loaded": _ALL_DATA_CACHE is not None,
        "kdtree_built": _KDTREE_CACHE is not None,
    }

    if _ALL_DATA_CACHE:
        status["num_dates"] = len(_ALL_DATA_CACHE)

    logger.debug(f"Health check: {status}")
    return JSONResponse(status)
