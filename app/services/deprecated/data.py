"""Data loading, and spatial indexing for DIC velocity fields."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from ..cache import cache
from ..config import get_settings
from .spatial import (
    nearest_node_grid,
    nearest_node_kdtree,
)

logger = logging.getLogger(__name__)
settings = get_settings()


###=== Reading DIC Files ===###
def read_dic_file(
    dic_file: Path,
    delimiter=",",
    skiprows: int = 0,
    invert_y: bool = False,
) -> dict[str, np.ndarray]:
    """Read DIC results: x, y, dx, dy, magnitude, ensamble_mad (optional).

    Args:
        dic_file: Path to the DIC file
        delimiter: Delimiter used in the file (default: comma)
        skiprows: Number of header rows to skip (default: 0)
        invert_y: Whether to invert Y coordinates (default: False)
    Returns:
        Dictionary with keys: "x", "y", "dx", "dy", "magnitude", "ensamble_mad"
    """

    if invert_y is None:
        invert_y = settings.invert_y

    logger.debug(f"Reading DIC file: {dic_file}")
    try:
        data = np.loadtxt(
            dic_file,
            delimiter=delimiter,
            skiprows=skiprows,
            dtype=np.float32,
            ndmin=2,
        )

        if data.size == 0:
            raise ValueError("Empty data")

        # Extract columns (views, not copies)
        x = data[:, 0]
        y = data[:, 1]
        dx = data[:, 2]
        dy = data[:, 3]

        if invert_y:
            y = -y
            dy = -dy

        # Compute magnitude from dx, dy columns
        magnitude = np.hypot(dx, dy)

        # Extract ensamble_mad (or zeros if not present)
        ensamble_mad = data[:, 4] if data.shape[1] > 4 else np.zeros_like(magnitude)

        dic_data = {
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "magnitude": magnitude,
            "ensamble_mad": ensamble_mad,
        }
        logger.debug(f"Loaded {len(x)} nodes from {dic_file.name}")

        return dic_data
    except Exception as e:
        logger.error(f"Failed to read DIC file {dic_file}: {e}")
        raise ValueError(f"Failed to read DIC file: {dic_file}") from e


def load_day_dic(
    data_dir: Path,
    date: str,
    file_pattern: str = "*.txt",
    filename_pattern: str | None = None,
) -> dict:
    """
    Fast version that uses preloaded data cache.

    Args:
        data_dir: Path to data directory
        date: Date string in YYYYMMDD format
        file_pattern: File pattern for DIC files
        filename_pattern: Pattern to extract date from filename

    Returns:
        Dictionary with DIC data for the specified date
    """
    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    if date not in all_data:
        logger.error(f"No data for date {date}")
        raise FileNotFoundError(f"No data for date {date}")

    return all_data[date]


def load_all_series(
    data_dir: Path,
    file_pattern: str | None = None,
    filename_pattern: str | None = None,
    node_x: float | None = None,
    node_y: float | None = None,
) -> dict:
    """
    Fast time series extraction using preloaded data and KDTree.

    Args:
        data_dir: Path to data directory (uses settings if None)
        file_pattern: File pattern for DIC files (uses settings if None)
        filename_pattern: Pattern to extract dates from filename (uses settings if None)
        node_x: X coordinate of the node
        node_y: Y coordinate of the node

    Returns:
        Dictionary with time series data
    """
    logger.info(f"Loading time series for node ({node_x}, {node_y})")

    if file_pattern is None:
        file_pattern = settings.file_pattern

    if filename_pattern is None:
        filename_pattern = settings.filename_pattern

    # Build KDTree and preload all data
    tree, coords = build_global_kdtree(data_dir, file_pattern, filename_pattern)
    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    # Find the nearest node index once
    dist, idx = tree.query([node_x, node_y], k=1)

    # Initialize result dictionary
    result = {
        "dates": [],
        "dt_hours": [],
        "dt_days": [],
        "dx": [],
        "dy": [],
        "disp_mag": [],
        "u": [],
        "v": [],
        "V": [],
        "ensamble_mad": [],
    }

    if not np.isfinite(dist):
        logger.warning(f"No valid node found for ({node_x}, {node_y})")
        return result

    # Extract time series for this index across all dates
    for date in sorted(all_data.keys()):
        data = all_data[date]
        result["dates"].append(date)
        result["dt_hours"].append(int(data["dt_hours"]))
        result["dt_days"].append(int(data["dt_days"]))
        result["dx"].append(float(data["dx"][idx]))
        result["dy"].append(float(data["dy"][idx]))
        result["disp_mag"].append(float(data["disp_mag"][idx]))
        result["u"].append(float(data["u"][idx]))
        result["v"].append(float(data["v"][idx]))
        result["V"].append(float(data["V"][idx]))
        result["ensamble_mad"].append(float(data["ensamble_mad"][idx]))

    logger.info(
        f"Loaded time series with {len(result['dates'])} points for node at "
        f"({coords[idx][0]:.1f}, {coords[idx][1]:.1f}), distance={dist:.2f}px"
    )

    return result


def preload_all_data(
    data_dir: Path | None = None,
    file_pattern: str | None = None,
    filename_pattern: str | None = None,
    dt_days_preferred: int | None = None,
    dt_days_tolerance: int = 1,
    force_reload: bool = False,
) -> dict[str, dict]:
    """
    Preload all DIC files into memory once for efficient time series queries.
    Always computes velocity in addition to displacement.

    Args:
        data_dir: Path to data directory (uses settings if None)
        file_pattern: File pattern for DIC files (uses settings if None)
        filename_pattern: Pattern to extract dates from filename (uses settings if None)
        dt_days_preferred: Preferred time difference in days (uses settings if None)
        dt_days_tolerance: Tolerance for dt_days filtering
        force_reload: If True, reload data even if cached

    Returns:
        Dictionary mapping date strings to data dictionaries
    """
    if cache.all_data is not None and not force_reload:
        logger.debug("Using cached data")
        return cache.all_data

    # Use settings as defaults
    data_dir = data_dir or settings.data_dir
    file_pattern = file_pattern or settings.file_pattern
    filename_pattern = filename_pattern or settings.filename_pattern
    dt_days_preferred = dt_days_preferred or settings.dt_days_preferred

    logger.info(f"Preloading all DIC files from {data_dir}")
    data_path = Path(data_dir)

    all_data = {}
    files = sorted(data_path.glob(file_pattern))

    # Select files based on preferred dt_days
    selected_files = []
    selected_dates_str = []
    selected_dt_hours = []

    for f in files:
        try:
            slave_date, master_date = parse_dic_filename(f, pattern=filename_pattern)
            date_str = slave_date.strftime(settings.date_format)

            dt_hours = int((slave_date - master_date).total_seconds() / 3600)
            if dt_hours <= 0:
                logger.warning(
                    f"Non-positive time difference in {f.name}, taking absolute value"
                )
                dt_hours = abs(dt_hours)

            # Filter by preferred dt_days
            dt_days = int(dt_hours / 24)
            if abs(dt_days - dt_days_preferred) <= dt_days_tolerance:
                selected_files.append(f)
                selected_dates_str.append(date_str)
                selected_dt_hours.append(dt_hours)

        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Skipping {f.name}: {e}")
            continue

    logger.info(
        f"Selected {len(selected_files)} files with dt_days≈{dt_days_preferred}"
    )

    # Load each selected file
    failed_files = []
    duplicate_dates = []

    for f, dt_hours, date_str in zip(
        selected_files, selected_dt_hours, selected_dates_str, strict=True
    ):
        try:
            # Check for duplicate dates
            if date_str in all_data:
                duplicate_dates.append((f.name, date_str))
                continue

            # Load the file
            dic_data = read_dic_file(f)

            # Displacement components
            dx = dic_data["dx"]
            dy = dic_data["dy"]
            disp_mag = dic_data["magnitude"]

            # Velocity components (always computed)
            if dt_hours > 24:
                u_vel = dx / dt_hours * 24.0  # Convert to per day
                v_vel = dy / dt_hours * 24.0
                V_vel = disp_mag / dt_hours * 24.0
            else:
                logger.warning(
                    f"dt_hours is {dt_hours} for {date_str}, using displacement as velocity"
                )
                u_vel = dx
                v_vel = dy
                V_vel = disp_mag

            # Store with date as key
            all_data[date_str] = {
                "x": dic_data["x"],
                "y": dic_data["y"],
                # Displacement data
                "dx": dx,
                "dy": dy,
                "disp_mag": disp_mag,
                # Velocity data
                "u": u_vel,
                "v": v_vel,
                "V": V_vel,
                # Metadata
                "ensamble_mad": dic_data["ensamble_mad"],
                "dt_hours": dt_hours,
                "dt_days": int(dt_hours / 24),
            }

        except (ValueError, FileNotFoundError) as e:
            failed_files.append((f.name, str(e)))
            logger.warning(f"Failed to load {f.name}: {e}")
            continue

    cache.all_data = all_data
    logger.info(f"Preloaded {len(all_data)} DIC files into cache")

    # Summary of issues
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files")
    if duplicate_dates:
        logger.warning(f"Skipped {len(duplicate_dates)} duplicate dates")

    return all_data


###=== Spatial Indexing


def build_global_kdtree(
    data_dir: Path | None = None,
    file_pattern: str | None = None,
    filename_pattern: str | None = None,
) -> tuple[cKDTree, np.ndarray]:
    """Build a KDTree from the first available date's coordinates."""
    if cache.kdtree is not None:
        logger.debug("Using cached KDTree")
        return cache.kdtree

    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    if not all_data:
        raise ValueError("No data available to build KDTree")

    first_date = next(iter(all_data.keys()))
    first_data = all_data[first_date]

    coords = np.column_stack([first_data["x"], first_data["y"]])

    # Optimize leafsize for data size
    leafsize = max(10, len(coords) // 1000)
    tree = cKDTree(coords, leafsize=leafsize, compact_nodes=True, balanced_tree=True)

    logger.info(f"Built optimized KDTree with {len(coords)} nodes, leafsize={leafsize}")
    cache.kdtree = (tree, coords)
    return tree, coords


def build_grid_index(
    data_dir: Path,
    file_pattern: str,
    filename_pattern: str | None = None,
) -> dict:
    """
    Build a grid-based spatial index for ultra-fast nearest neighbor lookup.
    Creates a hash map: {(grid_x, grid_y): [node_indices]}

    This is MUCH faster than KDTree for repeated queries when coordinates
    are discrete or on a regular grid.
    """
    if cache.grid is not None:
        logger.debug("Using cached grid index")
        return cache.grid

    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    if not all_data:
        raise ValueError("No data available to build grid index")

    # Use the first date to build the index (assumes same grid for all dates)
    first_date = next(iter(all_data.keys()))
    first_data = all_data[first_date]

    x = first_data["x"]
    y = first_data["y"]

    # Determine grid spacing (assume regular grid)
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_spacing = np.min(np.diff(x_unique)) if len(x_unique) > 1 else 1.0
    y_spacing = np.min(np.diff(y_unique)) if len(y_unique) > 1 else 1.0

    # Build hash map: {(grid_x, grid_y): [indices]}
    grid_map = {}
    for idx, (xi, yi) in enumerate(zip(x, y, strict=True)):
        # Snap to grid
        grid_x = int(np.round(xi / x_spacing))
        grid_y = int(np.round(yi / y_spacing))
        key = (grid_x, grid_y)

        if key not in grid_map:
            grid_map[key] = []
        grid_map[key].append(idx)

    grid_cache = {
        "map": grid_map,
        "x_spacing": x_spacing,
        "y_spacing": y_spacing,
        "coords": np.column_stack([x, y]),
        "x_bounds": (float(x.min()), float(x.max())),
        "y_bounds": (float(y.min()), float(y.max())),
    }

    cache.grid = grid_cache
    logger.info(
        f"Built grid index with {len(grid_map)} cells, "
        f"spacing=({x_spacing:.2f}, {y_spacing:.2f})"
    )
    return grid_cache


def nearest_node(
    x: float,
    y: float,
    date: str,
    *,
    data_dir: Path | None = None,
    file_pattern: str | None = None,
    filename_pattern: str | None = None,
    radius: float = 10.0,
    method: Literal["hybrid", "kdtree", "grid"] = "hybrid",
) -> dict | None:
    """
    High-level nearest-node lookup. Preloads data once and builds required index
    only once. Delegates to small helper functions.

    Raises:
        ValueError for unknown method.
    """
    logger.debug(
        "nearest_node: query (%s,%s) date=%s method=%s radius=%s",
        x,
        y,
        date,
        method,
        radius,
    )

    # Resolve defaults and preload
    data_dir = data_dir or settings.data_dir
    file_pattern = file_pattern or settings.file_pattern
    filename_pattern = filename_pattern or settings.filename_pattern

    # Load all data once (from cache if available)
    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)
    if date not in all_data:
        logger.debug("nearest_node: date %s not found in preloaded data", date)
        return None

    # Search using requested method
    if method == "kdtree":
        tree, coords = build_global_kdtree(data_dir, file_pattern, filename_pattern)
        return nearest_node_kdtree(tree, coords, all_data, date, x, y, radius)

    if method == "grid":
        grid_cache = build_grid_index(data_dir, file_pattern, filename_pattern)
        return nearest_node_grid(grid_cache, all_data, date, x, y, radius)

    if method == "hybrid":
        # prefer grid for small radius
        if radius <= 100:
            grid_cache = build_grid_index(data_dir, file_pattern, filename_pattern)
            return nearest_node_grid(grid_cache, all_data, date, x, y, radius)

        else:
            tree, coords = build_global_kdtree(data_dir, file_pattern, filename_pattern)
            return nearest_node_kdtree(tree, coords, all_data, date, x, y, radius)

    raise ValueError(f"Unknown method '{method}' - expected 'hybrid'|'kdtree'|'grid'")


###=== Utility Functions ===###
def _get_compiled_pattern(pattern: str) -> re.Pattern:
    """Cache compiled regex patterns."""
    return re.compile(pattern)


def format_date_for_display(
    date_str: str, input_format: str = "%Y%m%d", output_format: str = "%d/%m/%Y"
) -> str:
    """Convert date string from one format to another."""
    try:
        dt = datetime.strptime(date_str, input_format)
        return dt.strftime(output_format)
    except Exception:
        return date_str


def parse_display_date(
    display_date: str, input_format: str = "%d/%m/%Y", output_format: str = "%Y%m%d"
) -> str:
    """Convert display date back to storage format."""
    try:
        dt = datetime.strptime(display_date, input_format)
        return dt.strftime(output_format)
    except Exception:
        return display_date


def parse_dic_filename(
    file_path: Path, pattern: str | None = None
) -> tuple[datetime, datetime]:
    """
    Parse DIC filename using regex pattern.

    Args:
        file_path: Path to the DIC file
        pattern: Regex pattern with two capture groups for slave and master dates.

    Returns:
        Tuple of (slave_date, master_date)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If filename doesn't match pattern or dates can't be parsed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"DIC filename {file_path} does not exist")

    if pattern is None:
        pattern = settings.filename_pattern
    compiled_pattern = _get_compiled_pattern(pattern)

    try:
        match = compiled_pattern.search(file_path.name)
        if not match:
            raise ValueError(
                f"Filename {file_path.name} does not match pattern {pattern or pattern}"
            )

        if len(match.groups()) < 2:
            raise ValueError(
                f"Pattern must have at least 2 capture groups, got {len(match.groups())}"
            )

        slave_str = match.group(1)
        master_str = match.group(2)

        slave_datetime = datetime.strptime(slave_str, settings.date_format)
        master_datetime = datetime.strptime(master_str, settings.date_format)

    except Exception as e:
        raise ValueError(f"Failed to parse DIC filename {file_path.name}") from e

    return (slave_datetime, master_datetime)
