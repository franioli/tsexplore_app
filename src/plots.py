import base64
import logging
from io import BytesIO
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)

# Global cache for the base64 image (computed once)
_CACHED_IMAGE_DATA: dict[str, tuple[str | None, tuple[int, int] | None]] = {}


def _get_cached_image(
    background_image_path: str | None,
) -> tuple[str | None, tuple[int, int] | None]:
    """
    Get cached base64 image data. Only computed once per path.
    Returns (data_url, (width, height)) or (None, None).

    Args:
        background_image_path: Path to background image file
    """
    if not background_image_path:
        logger.debug("No background image path provided")
        return (None, None)

    # Return cached if available
    if background_image_path in _CACHED_IMAGE_DATA:
        logger.debug(f"Using cached image: {background_image_path}")
        return _CACHED_IMAGE_DATA[background_image_path]

    # Load and cache the image
    try:
        logger.info(f"Loading and caching background image: {background_image_path}")
        img = Image.open(background_image_path)
        w, h = img.size

        # Optionally resize if too large (faster rendering)
        max_dimension = 2048
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h

        bio = BytesIO()
        img.save(bio, format="PNG", optimize=True)
        bio.seek(0)
        b64 = base64.b64encode(bio.read()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        result = (data_url, (w, h))
        _CACHED_IMAGE_DATA[background_image_path] = result
        logger.info(f"Cached background image: {w}x{h}, {len(b64) / 1024:.1f} KB")
        return result

    except FileNotFoundError:
        logger.warning(f"Background image file not found: {background_image_path}")
        return (None, None)
    except Exception as e:
        logger.error(f"Failed to load background image: {e}")
        return (None, None)


def _create_raster_grid(x, y, values):
    """Create a regular grid from scattered points for raster visualization."""
    from scipy.interpolate import griddata

    logger.debug("Creating raster grid")
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    if len(x_unique) < 2 or len(y_unique) < 2:
        logger.warning("Not enough unique points for raster grid")
        return None, None, None

    # Create meshgrid
    xi = np.linspace(x.min(), x.max(), len(x_unique))
    yi = np.linspace(y.min(), y.max(), len(y_unique))
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate values onto grid
    Zi = griddata((x, y), values, (Xi, Yi), method="nearest")
    logger.debug(f"Created grid: {Xi.shape}")

    return Xi, Yi, Zi


def make_velocity_map_figure(
    x: np.ndarray,
    y: np.ndarray,
    mag: np.ndarray,
    std: np.ndarray | None = None,
    plot_type: Literal["scatter", "raster"] = "scatter",
    cmin=None,
    cmax=None,
    colorscale="Viridis",
    marker_size=6,
    marker_opacity=0.7,
    downsample_points=5000,
    selected_node=None,
    units="velocity (px/day)",
    background_image_path: str | None = None,
) -> go.Figure:
    """
    Create scatter or raster velocity/displacement plot.
    Data selection and units already handled by caller.

    Args:
        x, y: Coordinate arrays
        mag: Magnitude array (velocity or displacement)
        u, v: Component arrays
        nmad: Metadata array
        plot_type: "scatter" or "raster"
        cmin, cmax: Color scale limits
        colorscale: Plotly colorscale name
        marker_size: Size of scatter markers
        marker_opacity: Opacity of markers
        downsample_points: Max points to display
        selected_node: Dict with x, y of selected node
        units: Label for data units
        background_image_path: Path to background image (optional)

    Returns:
        Plotly Figure object
    """
    if plot_type not in {"scatter", "raster"}:
        logger.error(f"Unknown plot type: {plot_type}")
        plot_type = "scatter"

    logger.info(f"Creating {plot_type} plot")
    fig = go.Figure()

    # Use cached image
    img_src, img_size = _get_cached_image(background_image_path)
    if img_src and img_size:
        img_w, img_h = img_size
        fig.add_layout_image(
            dict(
                source=img_src,
                xref="x",
                yref="y",
                x=0,
                y=img_h,
                sizex=img_w,
                sizey=img_h,
                sizing="stretch",
                opacity=0.8,
                layer="below",
            )
        )

    if plot_type == "scatter":
        n_points = len(x)
        if n_points > downsample_points:
            logger.info(f"Downsampling from {n_points} to {downsample_points} points")
            step = max(1, n_points // downsample_points)
            x_plot = x[::step]
            y_plot = y[::step]
            mag_plot = mag[::step]
        else:
            x_plot = x
            y_plot = y
            mag_plot = mag

        fig.add_trace(
            go.Scattergl(
                x=x_plot,
                y=y_plot,
                mode="markers",
                marker=dict(
                    color=mag_plot,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title=f"|v|<br>{units}"),
                    size=marker_size,
                    opacity=marker_opacity,
                    line=dict(width=0),
                ),
                hovertemplate="x=%{x:.0f}<br>y=%{y:.0f}<br>|v|=%{marker.color:.3f}<extra></extra>",
                name="magnitude",
                showlegend=False,
            )
        )

    elif plot_type == "raster":
        Xi, Yi, Zi = _create_raster_grid(x, y, mag)
        if Xi is not None and Yi is not None and Zi is not None:
            fig.add_trace(
                go.Heatmap(
                    x=Xi[0, :],
                    y=Yi[:, 0],
                    z=Zi,
                    colorscale=colorscale,
                    zmin=cmin,
                    zmax=cmax,
                    colorbar=dict(title=f"|v|<br>{units}"),
                    hovertemplate="x=%{x:.0f}<br>y=%{y:.0f}<br>|v|=%{z:.2f}<extra></extra>",
                    name="magnitude",
                    opacity=0.6,
                )
            )
        else:
            logger.warning("Raster grid could not be created; no raster trace added")
    else:
        logger.error(f"Unsupported plot type: {plot_type}")

    if selected_node:
        fig.add_trace(
            go.Scatter(
                x=[selected_node["x"]],
                y=[selected_node["y"]],
                mode="markers",
                marker=dict(
                    size=16,
                    color="red",
                    symbol="x",
                    line=dict(width=2, color="white"),
                    opacity=0.9,
                ),
                name="Selected",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if img_size:
        img_w, img_h = img_size
        x_range = [0, img_w]
        y_range = [img_h, 0]
    else:
        x_range = [float(np.min(x)), float(np.max(x))]
        y_range = [float(np.max(y)), float(np.min(y))]

    fig.update_layout(
        title=dict(text=f"Velocity Field - {units}", font=dict(size=16)),
        xaxis=dict(
            range=x_range,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
        ),
        yaxis=dict(range=y_range, showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
        dragmode="pan",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def make_timeseries_figure(
    dates: list,
    u: np.ndarray,
    v: np.ndarray,
    V: np.ndarray,
    components=None,
    marker_mode="lines+markers",
    node_coords=None,
    y_label="Velocity (px/day)",
    xmin_date=None,
    xmax_date=None,
    ymin=None,
    ymax=None,
) -> go.Figure:
    """
    Create time series plot. Data selection and units already handled by caller.

    Args:
        dates: List of dates
        u, v, V: Component arrays (already in correct units)
        components: List of components to plot (e.g., ["u", "v", "V"])
        marker_mode: "lines+markers", "lines", or "markers"
        node_coords: Dict with "x", "y" of node
        y_label: Y-axis label
        xmin_date, xmax_date: Date range limits
        ymin, ymax: Y-axis limits

    Returns:
        Plotly Figure object
    """

    logger.info("Creating time series plot")
    fig = go.Figure()

    # Get label of the components to plot
    components = components or ["V"]

    # Add traces for selected components
    if "u" in components:
        fig.add_trace(
            go.Scattergl(
                x=dates,
                y=u,
                mode=marker_mode,
                name="u (East)",
                line=dict(color="blue", width=2),
            )
        )
    if "v" in components:
        fig.add_trace(
            go.Scattergl(
                x=dates,
                y=v,
                mode=marker_mode,
                name="v (North)",
                line=dict(color="green", width=2),
            )
        )
    if "V" in components:
        fig.add_trace(
            go.Scattergl(
                x=dates,
                y=V,
                mode=marker_mode,
                name="|v|",
                line=dict(color="red", width=2),
            )
        )

    title = "Time Series"
    if node_coords:
        title += f" - Node: ({node_coords['x']:.1f}, {node_coords['y']:.1f})"

    # Apply axis limits
    x_range = [xmin_date, xmax_date] if xmin_date and xmax_date else None
    y_range = [ymin, ymax] if ymin is not None and ymax is not None else None

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title=y_label,
        xaxis=dict(range=x_range, tickformat="%d/%m/%Y", tickangle=-45),
        yaxis=dict(range=y_range),
        dragmode="zoom",
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
    )

    return fig
