import base64
import logging
from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from typing import Any, Literal

import numpy as np
from PIL import Image
from plotly import graph_objects as go

from .data import cache

logger = logging.getLogger(__name__)


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
    cmin: float | None = None,
    cmax: float | None = None,
    colorscale: str = "Viridis",
    marker_size: int = 4,
    marker_opacity: float = 1.0,
    downsample_points: int = 5000,
    selected_node: dict[str, float] | None = None,
    units: str = "velocity (px/day)",
    background_image_path: str | None = None,
    background_opacity: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> go.Figure:
    """
    Create scatter or raster velocity/displacement plot.
    Data selection and units already handled by caller.

    Args:
        x: X coordinate array
        y: Y coordinate array
        mag: Magnitude array (velocity or displacement)
        std: Standard deviation array (optional, not yet implemented)
        plot_type: "scatter" or "raster"
        cmin: Minimum color scale value
        cmax: Maximum color scale value
        colorscale: Plotly colorscale name
        marker_size: Size of scatter markers
        marker_opacity: Opacity of markers (0-1)
        downsample_points: Maximum number of points to display
        selected_node: Dictionary with "x" and "y" keys for selected node
        units: Label for data units (e.g., "velocity (px/day)")
        background_image_path: Path to background image file
        metadata: Dictionary of metadata to display on plot. Keys are labels,
                 values are formatted automatically. Example:
                 {"Date": "2021-08-05", "dt": 3, "Mean velocity": 1.234,
                  "Std": 0.456, "N points": 10000}

    Returns:
        Plotly Figure object with velocity/displacement map

    Example:
        >>> metadata = {
        ...     "Date": "2021-08-05",
        ...     "dt (days)": 3,
        ...     "Mean |v|": 1.234,
        ...     "Std |v|": 0.456,
        ...     "N points": 10000,
        ... }
        >>> fig = make_velocity_map_figure(x, y, mag, metadata=metadata)
    """
    if plot_type not in {"scatter", "raster"}:
        logger.error(f"Unknown plot type: {plot_type}")
        plot_type = "scatter"

    logger.info(f"Creating {plot_type} plot with {len(x)} points")
    fig = go.Figure()

    if std is not None:
        logger.warning(
            "Standard deviation visualization not implemented in this version"
        )

    # Optional cached background image (downscaled for speed, stretched to data extents)
    img_src, img_size = _get_cached_image(background_image_path, max_dimension=2048)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if img_src and img_size:
        fig.add_layout_image(
            dict(
                source=img_src,
                xref="x",
                yref="y",
                x=x_min,
                y=y_max,  # top-left corner (y is reversed)
                sizex=x_max - x_min,
                sizey=y_max - y_min,
                sizing="stretch",
                opacity=background_opacity,
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

    # Build title with optional metadata
    title_text = f"Velocity Field - {units}"

    # Add metadata as annotation if provided
    if metadata:
        metadata_text = _format_metadata(metadata)

        # Add metadata annotation in top-right corner
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            text=metadata_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=8,
            font=dict(size=11, family="monospace", color="black"),
            align="left",
        )

    # Axis ranges: full data extents; always reverse Y for image compatibility
    x_range = [x_min, x_max]
    y_range = [y_min, y_max]
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        xaxis=dict(
            range=x_range,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
        ),
        yaxis=dict(range=y_range, autorange="min reversed", showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
        dragmode="pan",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def _format_metadata(metadata: dict[str, Any]) -> str:
    """
    Format metadata dictionary into a nice text annotation.

    Args:
        metadata: Dictionary with metadata key-value pairs

    Returns:
        Formatted HTML string for annotation

    Example:
        >>> meta = {"Date": "2021-08-05", "dt": 3, "Mean": 1.234}
        >>> _format_metadata(meta)
        '<b>Date:</b> 2021-08-05<br><b>dt:</b> 3<br><b>Mean:</b> 1.23'
    """
    lines = []

    for key, value in metadata.items():
        # Format value based on type
        if isinstance(value, float):
            # Format floats with 2-3 decimal places
            if abs(value) < 0.01 or abs(value) > 1000:
                formatted_value = f"{value:.2e}"  # Scientific notation
            else:
                formatted_value = f"{value:.3f}"
        elif isinstance(value, int):
            # Format integers with thousand separators
            formatted_value = f"{value:,}"
        elif isinstance(value, (list, tuple)):
            # Format sequences
            formatted_value = ", ".join(str(v) for v in value)
        else:
            # String or other types
            formatted_value = str(value)

        lines.append(f"<b>{key}:</b> {formatted_value}")

    return "<br>".join(lines)


def make_timeseries_figure(
    dates: list[datetime],
    u: np.ndarray,
    v: np.ndarray,
    V: np.ndarray,
    u_std: np.ndarray | None = None,
    v_std: np.ndarray | None = None,
    V_std: np.ndarray | None = None,
    components: Sequence[str] | None = None,
    marker_mode: Literal["lines+markers", "lines", "markers"] = "lines+markers",
    y_label: str = "Value",
    node_coords: dict[str, float] | None = None,
    xmin_date: str | None = None,
    xmax_date: str | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> go.Figure:
    """
    Create time series plot with optional error bands.
    Data selection and units already handled by caller.

    Args:
        dates: List of date datetime objects (e.g., [datetime(2021, 8, 5), datetime(2021, 8, 8), ...])
        u: East component array (velocity or displacement)
        v: North component array (velocity or displacement)
        V: Magnitude array (velocity or displacement)
        u_std: Standard deviation of u component (optional)
        v_std: Standard deviation of v component (optional)
        V_std: Standard deviation of magnitude (optional)
        components: Sequence of component names to plot.
                   Valid values: "u", "v", "V". Default: ["V"]
                   Example: ["u", "v"] or ("V",) or {"u", "V"}
        marker_mode: Display mode - "lines+markers", "lines", or "markers"
        y_label: Y-axis label (e.g., "Velocity (px/day)")
        node_coords: Dictionary with "x" and "y" keys for node coordinates
        xmin_date: Minimum date for x-axis (ISO format or display format)
        xmax_date: Maximum date for x-axis (ISO format or display format)
        ymin: Minimum value for y-axis
        ymax: Maximum value for y-axis
        metadata: Dictionary of metadata to display on plot. Keys are labels,
                 values are formatted automatically. Example:
                 {"Node": "(1234.5, 5678.9)", "Mean |v|": 1.234}

    Returns:
        Plotly Figure object with time series plot

    Raises:
        ValueError: If standard deviation arrays don't match data array sizes

    Example:
        >>> dates = ["2021-08-05", "2021-08-08", "2021-08-11"]
        >>> u = np.array([1.2, 1.5, 1.8])
        >>> V = np.array([2.1, 2.3, 2.5])
        >>> V_std = np.array([0.1, 0.15, 0.12])
        >>> fig = make_timeseries_figure(
        ...     dates,
        ...     u,
        ...     v,
        ...     V,
        ...     V_std=V_std,
        ...     components=["V"],
        ...     y_label="Velocity (px/day)",
        ... )
    """
    logger.info(f"Creating time series plot with {len(dates)} dates")

    if not dates:
        logger.warning("No dates provided for time series")
        return go.Figure()

    # Validate data arrays have same length
    n_points = len(dates)
    if len(u) != n_points or len(v) != n_points or len(V) != n_points:
        raise ValueError(
            f"Data arrays must have same length as dates ({n_points}). "
            f"Got: u={len(u)}, v={len(v)}, V={len(V)}"
        )

    # Validate standard deviation arrays if provided
    if u_std is not None and len(u_std) != n_points:
        raise ValueError(
            f"u_std must have same length as dates ({n_points}), got {len(u_std)}"
        )
    if v_std is not None and len(v_std) != n_points:
        raise ValueError(
            f"v_std must have same length as dates ({n_points}), got {len(v_std)}"
        )
    if V_std is not None and len(V_std) != n_points:
        raise ValueError(
            f"V_std must have same length as dates ({n_points}), got {len(V_std)}"
        )

    fig = go.Figure()

    # Validate component names
    components = list(components) if components is not None else ["V"]
    valid_components = {"u", "v", "V"}
    invalid = set(components) - valid_components
    if invalid:
        logger.warning(f"Invalid component names: {invalid}. Valid: {valid_components}")
        components = [c for c in components if c in valid_components]

    # Color mapping for components
    component_colors = {
        "u": "blue",
        "v": "green",
        "V": "red",
    }
    component_names = {
        "u": "u (East)",
        "v": "v (North)",
        "V": "|v|",
    }

    # Add traces for selected components with error bands
    if "u" in components:
        _add_trace_with_error_band(
            fig=fig,
            x=dates,
            y=u,
            y_std=u_std,
            name=component_names["u"],
            color=component_colors["u"],
            marker_mode=marker_mode,
        )

    if "v" in components:
        _add_trace_with_error_band(
            fig=fig,
            x=dates,
            y=v,
            y_std=v_std,
            name=component_names["v"],
            color=component_colors["v"],
            marker_mode=marker_mode,
        )

    if "V" in components:
        _add_trace_with_error_band(
            fig=fig,
            x=dates,
            y=V,
            y_std=V_std,
            name=component_names["V"],
            color=component_colors["V"],
            marker_mode=marker_mode,
        )

    # Build title
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

    # Add metadata as annotation if provided
    if metadata:
        metadata_text = _format_metadata(metadata)

        # Add metadata annotation in top-right corner
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.98,
            xanchor="right",
            yanchor="top",
            text=metadata_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=8,
            font=dict(size=11, family="monospace", color="black"),
            align="left",
        )

    return fig


def _add_trace_with_error_band(
    fig: go.Figure,
    x: list[datetime],
    y: np.ndarray,
    y_std: np.ndarray | None,
    name: str,
    color: str,
    marker_mode: str,
) -> None:
    """
    Add a trace with optional error band (±1 std) to a figure.

    Args:
        fig: Plotly Figure object to add trace to
        x: X-axis values (dates)
        y: Y-axis values (data)
        y_std: Standard deviation array (optional)
        name: Trace name for legend
        color: Color for line and fill
        marker_mode: Display mode - "lines+markers", "lines", or "markers"
    """
    # Add error band if std is provided
    if y_std is not None:
        # Upper bound
        y_upper = y + y_std
        # Lower bound
        y_lower = y - y_std

        # Add filled area for error band (±1 std)
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],  # x, then x reversed
                y=np.concatenate(
                    [y_upper, y_lower[::-1]]
                ),  # upper, then lower reversed
                fill="toself",
                fillcolor=_rgba_from_name(color, alpha=0.1),
                line=dict(color="rgba(255,255,255,0)"),  # Transparent line
                hoverinfo="skip",
                showlegend=False,
                name=f"{name} ±1σ",
            )
        )

    # Add main line trace
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode=marker_mode,
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        )
    )


def _rgba_from_name(color_name: str, alpha: float = 1.0) -> str:
    """
    Convert color name to rgba string with specified alpha.

    Args:
        color_name: Color name (e.g., "blue", "red", "green")
        alpha: Alpha/opacity value (0-1)

    Returns:
        RGBA color string (e.g., "rgba(0, 0, 255, 0.2)")
    """
    color_map = {
        "blue": "0, 0, 255",
        "red": "255, 0, 0",
        "green": "0, 128, 0",
        "orange": "255, 165, 0",
        "purple": "128, 0, 128",
        "cyan": "0, 255, 255",
        "magenta": "255, 0, 255",
        "yellow": "255, 255, 0",
    }

    rgb = color_map.get(color_name.lower(), "128, 128, 128")  # Default to gray
    return f"rgba({rgb}, {alpha})"
