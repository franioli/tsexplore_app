"""Spatial Indexing for Nearest Neighbor Search"""

import logging

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

logger = logging.getLogger(__name__)


def nearest_node_kdtree(
    tree: cKDTree,
    coords: np.ndarray,
    all_data: dict,
    date: str,
    x: float,
    y: float,
    radius: float = 50.0,
) -> dict | None:
    """Find nearest node using an already-built KDTree and preloaded data."""
    logger.debug("nearest_node_kdtree: query at (%s, %s) radius=%s", x, y, radius)

    if date not in all_data:
        logger.debug("nearest_node_kdtree: date %s not in preloaded data", date)
        return None

    dist, idx = tree.query([x, y], k=1, distance_upper_bound=radius)
    if not np.isfinite(dist) or idx >= len(coords):
        logger.debug("nearest_node_kdtree: no node within radius %s", radius)
        return None

    return _extract_node_data(all_data[date], int(idx))


def nearest_node_grid(
    grid_cache: dict,
    all_data: dict,
    date: str,
    x: float,
    y: float,
    radius: float = 50.0,
) -> dict | None:
    """Find nearest node using a pre-built grid index and preloaded data."""
    logger.debug("nearest_node_grid: query at (%s, %s) radius=%s", x, y, radius)

    if date not in all_data:
        logger.debug("nearest_node_grid: date %s not in preloaded data", date)
        return None

    grid_map = grid_cache["map"]
    x_spacing = grid_cache["x_spacing"]
    y_spacing = grid_cache["y_spacing"]
    coords = grid_cache["coords"]

    grid_x = int(np.round(x / x_spacing))
    grid_y = int(np.round(y / y_spacing))

    max_search_cells = int(np.ceil(radius / min(x_spacing, y_spacing))) + 1

    best_dist = np.inf
    best_idx = None

    for search_radius in range(max_search_cells + 1):
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                # only examine the ring for efficiency (skip inner cells duplicated earlier)
                if abs(dx) < search_radius and abs(dy) < search_radius:
                    continue

                key = (grid_x + dx, grid_y + dy)
                if key not in grid_map:
                    continue

                for idx in grid_map[key]:
                    node_x, node_y = coords[idx]
                    dist = np.hypot(x - node_x, y - node_y)
                    if dist <= radius and dist < best_dist:
                        best_dist = dist
                        best_idx = idx

        if best_idx is not None and search_radius > 0:
            break

    if best_idx is None:
        logger.debug("nearest_node_grid: no node found within radius %s", radius)
        return None

    return _extract_node_data(all_data[date], int(best_idx))


def _extract_node_data(data: dict, idx: int) -> dict:
    """Helper to extract data at a specific index."""
    result = {}
    for k, v in data.items():
        if k in ("dt_days", "dt_hours"):
            result[k] = v
            continue

        val = v[idx]
        if isinstance(val, np.floating):
            result[k] = float(val)
        elif isinstance(val, np.integer):
            result[k] = int(val)
        else:
            result[k] = val
    return result
