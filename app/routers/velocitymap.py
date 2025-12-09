import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..cache import get_loaded_dates
from ..config import get_logger, get_settings
from ..services.data import format_date_for_display, load_day_dic
from ..services.plots import make_velocity_map_figure

logger = get_logger()
settings = get_settings()

router = APIRouter()


@router.get("/dates")
async def dates_list():
    """Return loaded dates as YYYYMMDD list."""
    return JSONResponse({"dates": get_loaded_dates()})


@router.get("/")
async def velocity_map(
    date: str = Query(..., description="Date in YYYYMMDD format"),
    use_velocity: bool = Query(True),
    cmin: float | None = Query(None, ge=0),
    cmax: float | None = Query(None, ge=0),
    colorscale: str = Query(settings.default_colorscale),
    plot_type: str = Query(settings.default_plot_type),
    marker_size: int = Query(settings.default_marker_size, ge=1, le=50),
    marker_opacity: float = Query(settings.default_marker_opacity, ge=0.0, le=1.0),
    downsample_points: int = Query(settings.default_downsample_points, ge=100),
    selected_x: float | None = Query(None),
    selected_y: float | None = Query(None),
):
    """Create map for a given date."""
    logger.info(f"map: date={date} use_velocity={use_velocity}")
    try:
        raw = load_day_dic(
            settings.data_dir,
            date,
            settings.file_pattern,
            settings.filename_pattern,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No data for date {date}")

    x, y = raw["x"], raw["y"]
    dt_days = raw.get("dt_days")
    mag = raw["V"] if use_velocity else raw["disp_mag"]
    units = "velocity (px/day)" if use_velocity else "displacement (px)"

    metadata = {
        "Date": format_date_for_display(date),
        "dt (days)": dt_days or "N/A",
        "N points": len(x),
        "Median |v|": float(np.median(mag)),
    }

    selected_node = (
        {"x": selected_x, "y": selected_y} if selected_x and selected_y else None
    )

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
        background_opacity=settings.background_image_opacity,
        metadata=metadata,
    )

    return JSONResponse(fig.to_dict())
