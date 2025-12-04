import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from scipy.spatial import cKDTree

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Load defaults but don't use as globals in functions
_DATA_DIR = os.getenv("DATA_DIR", "./data/day_dic")
_FILE_PATTERN = os.getenv("FILE_PATTERN", "*.txt")
_FILENAME_PATTERN = os.getenv("FILENAME_PATTERN", r"day_dic_(\d{8})-(\d{8})")
_INVERT_Y = os.getenv("INVERT_Y", "false").lower() == "true"
DATE_FORMAT = os.getenv("DATE_FORMAT", "%Y%m%d")

# Global cache for data
_GRID_CACHE: dict | None = None  # Cache for grid-based lookup
_ALL_DATA_CACHE: dict[str, dict] | None = None
_VELOCITY_CACHE: dict[tuple[str, bool], dict] = {}
_KDTREE_CACHE: tuple[cKDTree, np.ndarray] | None = None
_COMPILED_PATTERN = re.compile(_FILENAME_PATTERN)

logger.info(
    f"Data module initialized: DATA_DIR={_DATA_DIR}, FILE_PATTERN={_FILE_PATTERN}, INVERT_Y={_INVERT_Y}"
)

###=== Configuration Accessors ===###


def get_data_dir() -> Path:
    """Return the configured data directory as a Path."""
    return Path(_DATA_DIR)


def get_file_pattern() -> str:
    """Return the configured file pattern."""
    return _FILE_PATTERN


def get_filename_pattern() -> str:
    """Return the configured filename regex pattern."""
    return _FILENAME_PATTERN


def get_invert_y() -> bool:
    """Return whether to invert Y coordinates."""
    return _INVERT_Y


@lru_cache(maxsize=128)
def _get_compiled_pattern(pattern: str) -> re.Pattern:
    """Cache compiled regex patterns."""
    return re.compile(pattern)


def get_loaded_dates() -> list[str]:
    """Get the list of actually loaded dates from the preloaded data."""
    from .data import _ALL_DATA_CACHE

    if _ALL_DATA_CACHE is None:
        return []

    return sorted(_ALL_DATA_CACHE.keys())


###=== Filename Parsing and Date Listing ===###


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
        compiled_pattern = _COMPILED_PATTERN
    else:
        compiled_pattern = _get_compiled_pattern(pattern)

    try:
        match = compiled_pattern.search(file_path.name)
        if not match:
            raise ValueError(
                f"Filename {file_path.name} does not match pattern {pattern or _FILENAME_PATTERN}"
            )

        if len(match.groups()) < 2:
            raise ValueError(
                f"Pattern must have at least 2 capture groups, got {len(match.groups())}"
            )

        slave_str = match.group(1)
        master_str = match.group(2)

        slave_datetime = datetime.strptime(slave_str, DATE_FORMAT)
        master_datetime = datetime.strptime(master_str, DATE_FORMAT)

    except Exception as e:
        raise ValueError(f"Failed to parse DIC filename {file_path.name}") from e

    return (slave_datetime, master_datetime)


###=== Reading DIC Files ===###
def read_dic_file(
    dic_file: Path,
    delimiter=",",
    skiprows: int = 0,
    invert_y: bool = False,
) -> dict[str, np.ndarray]:
    """Read DIC results: x, y, dx, dy, magnitude, nmad (optional).

    Args:
        dic_file: Path to the DIC file
        delimiter: Delimiter used in the file (default: comma)
        skiprows: Number of header rows to skip (default: 0)
        invert_y: Whether to invert Y coordinates (default: False)
    Returns:
        Dictionary with keys: "x", "y", "dx", "dy", "magnitude", "nmad"
    """
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

        # Extract NMAD (or zeros if not present)
        if data.shape[1] > 4:
            nmad = data[:, 4]
        else:
            nmad = np.zeros_like(magnitude)

        logger.debug(f"Loaded {len(x)} nodes from {dic_file.name}")

        dic_data = {
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "magnitude": magnitude,
            "nmad": nmad,
        }

        return dic_data
    except Exception as e:
        logger.error(f"Failed to read DIC file {dic_file}: {e}")
        raise ValueError(f"Failed to read DIC file: {dic_file}") from e


def preload_all_data(
    data_dir: Path,
    file_pattern: str,
    filename_pattern: str | None = None,
    dt_days_preferred: int | None = None,
    dt_days_tolerance: int = 1,
    force_reload: bool = False,
) -> dict[str, dict]:
    """
    Preload all DIC files into memory once for efficient time series queries.
    Always computes velocity in addition to displacement.

    Args:
        data_dir: Path to data directory
        file_pattern: File pattern for DIC files
        filename_pattern: Pattern to extract dates from filename
        force_reload: If True, reload data even if cached

    Returns:
        Dictionary mapping date strings to data dictionaries with keys:
            "x", "y", "dx", "dy", "disp_mag", "u", "v", "V", "nmad", "dt_hours", "dt_days"

    """
    global _ALL_DATA_CACHE

    if _ALL_DATA_CACHE is not None and not force_reload:
        logger.debug("Using cached data")
        return _ALL_DATA_CACHE

    logger.info(f"Preloading all DIC files from {data_dir}")
    data_path = Path(data_dir)
    invert_y = get_invert_y()

    all_data = {}
    files = sorted(data_path.glob(file_pattern))

    # Select files based on preferred dt_days if specified
    selected_files = []
    selected_dates_str = []
    selected_dt_hours = []
    for f in files:
        try:
            slave_date, master_date = parse_dic_filename(f, pattern=filename_pattern)
            date_str = slave_date.strftime(DATE_FORMAT)

            dt_hours = int((slave_date - master_date).total_seconds() / 3600)
            if dt_hours <= 0:
                logger.warning(
                    "Non-positive time difference, taking absolute value. Result may be invalid."
                )
                dt_hours = abs(dt_hours)

            # Filter by preferred dt_days if specified
            if dt_days_preferred is not None:
                dt_days = int(dt_hours / 24)
                if abs(dt_days - dt_days_preferred) <= dt_days_tolerance:
                    selected_files.append(f)
                    selected_dates_str.append(date_str)
                    selected_dt_hours.append(dt_hours)

            else:  # Otherwise select all files
                selected_files.append(f)
                selected_dates_str.append(date_str)
                selected_dt_hours.append(dt_hours)

        except (ValueError, FileNotFoundError):
            continue

    files = selected_files
    dt_hours_list = selected_dt_hours
    date_str_list = selected_dates_str
    logger.info(f"Selected {len(files)} files with dt_days={dt_days_preferred}")

    # Load each selected file
    failed_files = []  # Track failed files
    duplicate_dates = []  # Track duplicate dates
    for f, dt_hours, date_str in zip(files, dt_hours_list, date_str_list, strict=True):
        try:
            # Check for duplicate dates
            if date_str in all_data:
                duplicate_dates.append((f.name, date_str))
                continue

            # Load the file
            dic_data = read_dic_file(f, invert_y=invert_y)

            # Displacement components
            dx = dic_data["dx"]
            dy = dic_data["dy"]
            disp_mag = dic_data["magnitude"]

            # Velocity components (always computed, even if not displayed)
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
                "nmad": dic_data["nmad"],
                "dt_hours": dt_hours,
                "dt_days": int(dt_hours / 24),
            }

        except (ValueError, FileNotFoundError) as e:
            failed_files.append((f.name, str(e)))
            logger.warning(f"Skipping file {f.name}: {e}")
            continue

    _ALL_DATA_CACHE = all_data
    logger.info(f"Preloaded {len(all_data)} DIC files.")

    # Summary of issues
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files:")
        for fname, error in failed_files:
            logger.warning(f"  - {fname}: {error}")

    if duplicate_dates:
        logger.warning(f"Skipped {len(duplicate_dates)} duplicate dates:")
        for fname, date in duplicate_dates:
            logger.warning(f"  - {fname} (date: {date})")

    return all_data


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
    file_pattern: str,
    filename_pattern: str | None,
    node_x: float,
    node_y: float,
) -> dict:
    """
    Fast time series extraction using preloaded data and KDTree.
    Args:
        data_dir: Path to data directory
        file_pattern: File pattern for DIC files
        filename_pattern: Pattern to extract dates from filename
        node_x: X coordinate of the node
        node_y: Y coordinate of the node
    Returns:
        Dictionary with time series data:
            "dates": list of date strings
            "u": list of u velocities
            "v": list of v velocities
            "V": list of velocity magnitudes
            "nmad": list of nmad values
            "dt_days": list of dt_days values
    """
    logger.info(f"Loading time series for node ({node_x}, {node_y})")

    # Build KDTree and preload all data
    tree, coords = build_global_kdtree(data_dir, file_pattern, filename_pattern)
    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    # Find the nearest node index once
    dist, idx = tree.query([node_x, node_y], k=1)

    if not np.isfinite(dist):
        logger.warning(f"No valid node found for ({node_x}, {node_y})")
        return {
            "dates": [],
            "u": [],
            "v": [],
            "V": [],
            "nmad": [],
            "dt_days": [],
        }

    # Extract time series for this index across all dates
    dates_out = []
    u_out = []
    v_out = []
    V_out = []
    nmad_out = []
    dt_days_out = []

    for date in sorted(all_data.keys()):
        data = all_data[date]

        dates_out.append(date)
        u_out.append(float(data["u"][idx]))
        v_out.append(float(data["v"][idx]))
        V_out.append(float(data["V"][idx]))
        nmad_out.append(float(data["nmad"][idx]))
        dt_days_out.append(int(data["dt_days"]))

    logger.info(f"Loaded time series with {len(dates_out)} points in O(1) time")

    return {
        "dates": dates_out,
        "u": u_out,
        "v": v_out,
        "V": V_out,
        "nmad": nmad_out,
        "dt_days": dt_days_out,
    }


###=== Spatial Indexing for Nearest Neighbor Search ===###
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
    global _GRID_CACHE

    if _GRID_CACHE is not None:
        logger.debug("Using cached grid index")
        return _GRID_CACHE

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

    if len(x_unique) > 1:
        x_spacing = np.min(np.diff(x_unique))
    else:
        x_spacing = 1.0

    if len(y_unique) > 1:
        y_spacing = np.min(np.diff(y_unique))
    else:
        y_spacing = 1.0

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

    _GRID_CACHE = {
        "map": grid_map,
        "x_spacing": x_spacing,
        "y_spacing": y_spacing,
        "coords": np.column_stack([x, y]),
        "x_bounds": (float(x.min()), float(x.max())),
        "y_bounds": (float(y.min()), float(y.max())),
    }

    logger.info(
        f"Built grid index with {len(grid_map)} cells, spacing=({x_spacing:.2f}, {y_spacing:.2f})"
    )
    return _GRID_CACHE


def build_global_kdtree(
    data_dir: Path,
    file_pattern: str,
    filename_pattern: str | None = None,
) -> tuple[cKDTree, np.ndarray]:
    """
    Build a KDTree from the first available date's coordinates.
    Optimized with leafsize parameter for faster queries.
    """
    global _KDTREE_CACHE

    if _KDTREE_CACHE is not None:
        logger.debug("Using cached KDTree")
        return _KDTREE_CACHE

    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    if not all_data:
        raise ValueError("No data available to build KDTree")

    first_date = next(iter(all_data.keys()))
    first_data = all_data[first_date]

    coords = np.column_stack([first_data["x"], first_data["y"]])

    # Optimize leafsize for your data size
    # Larger leafsize = faster build, slightly slower query
    # Smaller leafsize = slower build, faster query
    leafsize = max(10, len(coords) // 1000)  # Adaptive leafsize
    tree = cKDTree(coords, leafsize=leafsize, compact_nodes=True, balanced_tree=True)

    logger.info(f"Built optimized KDTree with {len(coords)} nodes, leafsize={leafsize}")
    _KDTREE_CACHE = (tree, coords)
    return tree, coords


def nearest_node_kdtree(
    data_dir: Path,
    file_pattern: str,
    filename_pattern: str | None,
    date: str,
    x: float,
    y: float,
    radius: float = 50.0,
) -> dict | None:
    """
    Find nearest node using KDTree spatial index.
    Fast O(log n) lookup, good for large datasets and variable query points.

    Args:
        data_dir: Path to data directory
        file_pattern: File pattern for DIC files
        filename_pattern: Pattern to extract date from filename
        date: Date in YYYYMMDD format
        x: X coordinate
        y: Y coordinate
        radius: Search radius in pixels

    Returns:
        Dictionary with node data or None if not found within radius
    """
    logger.debug(
        f"Finding nearest node using KDTree at ({x}, {y}) within radius {radius}"
    )

    try:
        tree, coords = build_global_kdtree(data_dir, file_pattern, filename_pattern)
        all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

        if date not in all_data:
            logger.warning(f"Date {date} not found in preloaded data")
            return None

        # Query KDTree for nearest neighbor within radius
        dist, idx = tree.query([x, y], k=1, distance_upper_bound=radius)

        # Check if a valid result was found
        if not np.isfinite(dist) or idx >= len(coords):
            logger.debug(f"No node found within radius {radius} of ({x}, {y})")
            return None

        # Extract data at the found index
        data = all_data[date]
        result = {}

        for k, v in data.items():
            # Scalar metadata fields - copy directly
            if k in ("dt_days", "dt_hours"):
                result[k] = v
                continue

            # Array fields - extract value at index
            val = v[idx]
            if isinstance(val, np.floating):
                result[k] = float(val)
            elif isinstance(val, np.integer):
                result[k] = int(val)
            else:
                result[k] = val

        logger.debug(
            f"Found node at ({result['x']:.2f}, {result['y']:.2f}), distance={dist:.2f}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in nearest_node_kdtree: {e}", exc_info=True)
        return None


def nearest_node_grid(
    data_dir: Path,
    file_pattern: str,
    filename_pattern: str | None,
    date: str,
    x: float,
    y: float,
    radius: float = 50.0,
) -> dict | None:
    """
    Ultra-fast nearest node search using grid-based spatial index.
    O(1) average case vs O(log n) for KDTree.
    """
    grid_cache = build_grid_index(data_dir, file_pattern, filename_pattern)
    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    if date not in all_data:
        logger.warning(f"Date {date} not found in preloaded data")
        return None

    grid_map = grid_cache["map"]
    x_spacing = grid_cache["x_spacing"]
    y_spacing = grid_cache["y_spacing"]
    coords = grid_cache["coords"]

    # Snap query point to grid
    grid_x = int(np.round(x / x_spacing))
    grid_y = int(np.round(y / y_spacing))

    # Search in expanding square around query point
    max_search_cells = int(np.ceil(radius / min(x_spacing, y_spacing))) + 1

    best_dist = np.inf
    best_idx = None

    for search_radius in range(max_search_cells + 1):
        found_in_ring = False

        # Search cells in a square ring
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                # Only search the outer ring
                if abs(dx) < search_radius and abs(dy) < search_radius:
                    continue

                key = (grid_x + dx, grid_y + dy)

                if key in grid_map:
                    # Check all nodes in this cell
                    for idx in grid_map[key]:
                        node_x, node_y = coords[idx]
                        dist = np.sqrt((x - node_x) ** 2 + (y - node_y) ** 2)

                        if dist < best_dist and dist <= radius:
                            best_dist = dist
                            best_idx = idx
                            found_in_ring = True

        # If we found a node and checked all closer cells, we can stop
        if found_in_ring and search_radius > 0:
            break

    if best_idx is None:
        logger.debug(f"No node found within radius {radius} of ({x}, {y})")
        return None

    # Extract values at this index
    data = all_data[date]
    result = {}
    for k, v in data.items():
        # Scalar metadata fields - copy directly
        if k in ("dt_days", "dt_hours"):
            result[k] = v
            continue

        # Array fields - extract value at index
        val = v[best_idx]
        if isinstance(val, np.floating):
            result[k] = float(val)
        elif isinstance(val, np.integer):
            result[k] = int(val)
        else:
            result[k] = val

    logger.debug(
        f"Found node at ({result['x']:.2f}, {result['y']:.2f}), distance={best_dist:.2f}"
    )
    return result


def nearest_node_hybrid(
    data_dir: Path,
    file_pattern: str,
    filename_pattern: str | None,
    date: str,
    x: float,
    y: float,
    radius: float = 50.0,
) -> dict | None:
    """
    Hybrid approach: use grid for small radius, KDTree for large radius.
    """
    # Use grid method for small radius (faster)
    if radius <= 100:
        return nearest_node_grid(
            data_dir, file_pattern, filename_pattern, date, x, y, radius
        )

    # Use KDTree for large radius (more reliable)
    tree, coords = build_global_kdtree(data_dir, file_pattern, filename_pattern)
    all_data = preload_all_data(data_dir, file_pattern, filename_pattern)

    if date not in all_data:
        return None

    dist, idx = tree.query([x, y], k=1, distance_upper_bound=radius)

    if not np.isfinite(dist) or idx >= len(coords):
        logger.debug(f"No node found within radius {radius} of ({x}, {y})")
        return None

    data = all_data[date]
    result = {}
    for k, v in data.items():
        # Scalar metadata fields - copy directly
        if k in ("dt_days", "dt_hours"):
            result[k] = v
            continue

        # Array fields - extract value at index
        val = v[idx]
        if isinstance(val, np.floating):
            result[k] = float(val)
        elif isinstance(val, np.integer):
            result[k] = int(val)
        else:
            result[k] = val

    logger.debug(
        f"Found node at ({result['x']:.2f}, {result['y']:.2f}), distance={dist:.2f}"
    )
    return result


def nearest_node(
    data_dir: Path,
    file_pattern: str,
    filename_pattern: str | None,
    date: str,
    x: float,
    y: float,
    radius: float = 10.0,
    method: Literal["hybrid", "kdtree", "grid"] = "hybrid",
) -> dict | None:
    """
    Find nearest node to (x, y) at given date.

    Args:
        data_dir: Path to data directory
        file_pattern: File pattern for DIC files
        filename_pattern: Pattern to extract date from filename
        date: Date in YYYYMMDD format
        x: X coordinate
        y: Y coordinate
        radius: Search radius in pixels
        method: Search method - "hybrid" (default), "kdtree", or "grid"

    Returns:
        Dictionary with "x", "y" keys or None if not found
    """
    logger.debug(f"Finding nearest node at ({x}, {y}) using {method} method")

    if method == "kdtree":
        return nearest_node_kdtree(
            data_dir, file_pattern, filename_pattern, date, x, y, radius
        )
    elif method == "grid":
        return nearest_node_grid(
            data_dir, file_pattern, filename_pattern, date, x, y, radius
        )
    elif method == "hybrid":
        return nearest_node_hybrid(
            data_dir, file_pattern, filename_pattern, date, x, y, radius
        )
    else:
        logger.warning(f"Unknown method '{method}', defaulting to 'hybrid'")
        return nearest_node_hybrid(
            data_dir, file_pattern, filename_pattern, date, x, y, radius
        )
