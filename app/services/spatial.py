"""Spatial Indexing for Nearest Neighbor Search"""

import logging
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from ..cache import cache
from .provider import DataProvider, get_data_provider

logger = logging.getLogger(__name__)


def build_kdtree(provider: DataProvider | None = None) -> tuple[cKDTree, np.ndarray]:
    """Build KDTree from coordinates."""
    if cache.kdtree is not None:
        return cache.kdtree

    if provider is None:
        provider = get_data_provider()

    coords = provider.get_coordinates()
    leafsize = max(10, len(coords) // 1000)
    tree = cKDTree(coords, leafsize=leafsize, compact_nodes=True, balanced_tree=True)

    logger.info(f"Built KDTree with {len(coords)} nodes, leafsize={leafsize}")
    cache.kdtree = (tree, coords)
    return tree, coords


def build_grid_index(provider: DataProvider | None = None) -> dict:
    """Build grid-based spatial index."""
    if cache.grid is not None:
        return cache.grid

    if provider is None:
        provider = get_data_provider()

    coords = provider.get_coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

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
        "coords": coords,
        "x_bounds": (float(x.min()), float(x.max())),
        "y_bounds": (float(y.min()), float(y.max())),
    }

    cache.grid = grid_cache
    return grid_cache


def nearest_node(
    x: float,
    y: float,
    date: str,
    *,
    radius: float = 10.0,
    method: Literal["hybrid", "kdtree", "grid"] = "hybrid",
    provider: DataProvider | None = None,
) -> dict | None:
    """
    High-level nearest-node lookup using provider pattern.

    Args:
        x: X coordinate
        y: Y coordinate
        date: Date in YYYYMMDD format
        radius: Search radius
        method: Search method ('hybrid', 'kdtree', 'grid')
        provider: Optional data provider (uses default if None)

    Returns:
        Dictionary with node data or None if not found
    """
    logger.debug(
        f"nearest_node: query ({x},{y}) date={date} method={method} radius={radius}"
    )

    if provider is None:
        provider = get_data_provider()

    # Load all data once
    all_data = provider.load_all()
    if date not in all_data:
        logger.debug(f"nearest_node: date {date} not found in preloaded data")
        return None

    # Search using requested method
    if method == "kdtree":
        tree, coords = build_kdtree(provider)
        return _nearest_node_kdtree(tree, coords, all_data, date, x, y, radius)

    if method == "grid":
        grid_cache = build_grid_index(provider)
        return _nearest_node_grid(grid_cache, all_data, date, x, y, radius)

    if method == "hybrid":
        # Prefer grid for small radius
        if radius <= 100:
            grid_cache = build_grid_index(provider)
            return _nearest_node_grid(grid_cache, all_data, date, x, y, radius)
        else:
            tree, coords = build_kdtree(provider)
            return _nearest_node_kdtree(tree, coords, all_data, date, x, y, radius)

    raise ValueError(f"Unknown method '{method}' - expected 'hybrid'|'kdtree'|'grid'")


###=== Backend functions ===###


def _nearest_node_kdtree(
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


def _nearest_node_grid(
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
