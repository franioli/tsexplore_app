import base64
from datetime import datetime
from io import BytesIO

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image

from ..cache import cache
from ..config import get_logger, get_settings
from ..services.data_provider import get_data_provider
from ..services.plots import make_velocity_map_figure

logger = get_logger()
settings = get_settings()

router = APIRouter()


@router.get("/dates")
async def dates_list():
    """Return loaded dates as YYYYMMDD list."""
    provider = get_data_provider()
    dates = provider.get_available_dates()
    return JSONResponse({"dates": dates})


def _get_cached_image(
    background_image_path: str | None,
    max_dimension: int | None = None,
) -> tuple[str | None, tuple[int, int] | None]:
    """
    Get cached base64 image data. Only computed once per path.
    Returns (data_url, (width, height)) or (None, None).

    The image can be optionally downscaled (max_dimension) to speed up plotting.
    """
    if not background_image_path:
        logger.debug("No background image path provided")
        return (None, None)

    # Try cache first
    data_url, size, raw_bytes = cache.get_image(background_image_path)
    if data_url and size:
        logger.debug(f"Using cached image: {background_image_path}")
        return data_url, size

    # Load and cache the image
    try:
        logger.info(f"Loading and caching background image: {background_image_path}")
        img = Image.open(background_image_path)
        w, h = img.size

        # Downscale for speed if too large
        if max_dimension and (w > max_dimension or h > max_dimension):
            scale = max_dimension / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h

        bio = BytesIO()
        img.save(bio, format="PNG", optimize=True)
        bio.seek(0)
        raw = bio.read()
        b64 = base64.b64encode(raw).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        # Store in cache (metadata + raw bytes)
        cache.store_image(background_image_path, data_url, (w, h), raw)
        logger.info(f"Cached background image: {w}x{h}, {len(b64) / 1024:.1f} KB")
        return data_url, (w, h)

    except FileNotFoundError:
        logger.warning(f"Background image file not found: {background_image_path}")
        return (None, None)
    except Exception as e:
        logger.error(f"Failed to load background image: {e}")
        return (None, None)


@router.get("/")
async def velocity_map(
    reference_date: str | None = Query(
        None,
        description=(
            "Reference date in YYYYMMDD format (FINAL image date used for the DIC computation)."
        ),
    ),
    dt_days: int | None = Query(
        None, ge=0, description="Select record by dt_days (exact)."
    ),
    prefer_dt_days: int | None = Query(None, ge=0, description="Pick closest dt_days."),
    prefer_dt_tolerance: int | None = Query(
        None, ge=0, description="Tolerance (days) for closest dt_days selection."
    ),
    use_velocity: bool = Query(True),
    cmin: float | None = Query(None, ge=0),
    cmax: float | None = Query(None, ge=0),
    colorscale: str = Query(settings.default_colorscale),
    plot_type: str = Query(settings.default_plot_type),
    marker_size: int = Query(settings.default_marker_size, ge=1, le=50),
    marker_opacity: str = Query(
        str(settings.default_marker_opacity), description="0..1 or 'auto'"
    ),
    downsample_points: int = Query(settings.default_downsample_points, ge=100),
    selected_x: float | None = Query(None),
    selected_y: float | None = Query(None),
):
    """Create map for a given date."""
    logger.info(f"map: date={reference_date} use_velocity={use_velocity}")

    # Get data provider
    provider = get_data_provider()

    # Fetch DIC data
    raw = provider.get_dic_data(
        reference_date,
        dt_days=dt_days,
        prefer_dt_days=prefer_dt_days,
        prefer_dt_tolerance=prefer_dt_tolerance,
    )
    if not raw:
        raise HTTPException(
            status_code=404,
            detail=(
                "No data for reference date "
                f"{reference_date} (dt_days={dt_days}, prefer_dt_days={prefer_dt_days}, "
                f"prefer_dt_tolerance={prefer_dt_tolerance})"
            ),
        )

    # Prepare data for plotting
    x, y = raw["x"], raw["y"]
    dt_days = raw.get("dt_days")
    mag = raw["V"] if use_velocity else raw["disp_mag"]
    units = "velocity (px/day)" if use_velocity else "displacement (px)"
    try:
        date_display = datetime.strptime(reference_date, "%Y%m%d").strftime("%d/%m/%Y")
    except ValueError:
        date_display = reference_date
    metadata = {
        "Date": date_display,
        "dt (days)": dt_days or "N/A",
        "N points": len(x),
        "Median |v|": float(np.median(mag)),
    }

    # Get selected node if any
    selected_node = (
        {"x": selected_x, "y": selected_y}
        if selected_x is not None and selected_y is not None
        else None
    )

    # Validate marker_opacity
    if isinstance(marker_opacity, str) and marker_opacity.lower() == "auto":
        marker_opacity_value = "auto"
    else:
        try:
            marker_opacity_value = float(marker_opacity)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail="marker_opacity must be a float in [0,1] or 'auto'",
            )
        if not (0.0 <= marker_opacity_value <= 1.0):
            raise HTTPException(
                status_code=422,
                detail="marker_opacity must be in [0,1] or 'auto'",
            )

    # Load/cached background here
    bg_src, _bg_size = _get_cached_image(settings.background_image, max_dimension=2048)

    # make figure
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
        marker_opacity=marker_opacity_value,  # <-- float or "auto"
        downsample_points=downsample_points,
        selected_node=selected_node,
        units=units,
        background_image=bg_src,  # <- pass prepared image (or None)
        background_opacity=settings.background_image_opacity,
        metadata=metadata,
    )

    return JSONResponse(fig.to_dict())
