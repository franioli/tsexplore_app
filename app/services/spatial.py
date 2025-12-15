"""Spatial indexing and nearest-neighbor utilities.

This module provides:
- A cached KDTree for fast nearest-node lookup.
- A cached grid index for efficient small-radius searches.
- A high-level `nearest_node()` helper that fetches the DIC payload for a given
  slave date via the active DataProvider, and returns values at the nearest node.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from ..cache import cache
from .data_provider import DataProvider, get_data_provider

logger = logging.getLogger(__name__)


def build_kdtree(provider: DataProvider | None = None) -> tuple[cKDTree, np.ndarray]:
    """Build (or reuse) a KDTree from the provider coordinates.

    Args:
        provider: Optional data provider. If None, uses the default provider.

    Returns:
        A tuple `(tree, coords)` where:
          - `tree` is a `scipy.spatial.cKDTree`
          - `coords` is an `(N, 2)` array of `[x, y]` coordinates
    """
    if cache.kdtree is not None:
        return cache.kdtree

    if provider is None:
        provider = get_data_provider()

    coords = provider.get_coordinates()
    leafsize = max(10, len(coords) // 1000)
    tree = cKDTree(coords, leafsize=leafsize, compact_nodes=True, balanced_tree=True)

    logger.info("Built KDTree with %s nodes, leafsize=%s", len(coords), leafsize)
    cache.kdtree = (tree, coords)
    return tree, coords


def build_grid_index(provider: DataProvider | None = None) -> dict[str, Any]:
    """Build (or reuse) a grid-based spatial index for nearest-node searches.

    The grid index maps discretized grid cells to point indices, assuming that
    the coordinate set forms (approximately) a regular grid.

    Args:
        provider: Optional data provider. If None, uses the default provider.

    Returns:
        A dict with keys:
          - `map`: dict[(grid_x, grid_y) -> list[int]]
          - `x_spacing`, `y_spacing`: inferred grid spacing
          - `coords`: `(N, 2)` coordinate array
          - `x_bounds`, `y_bounds`: min/max bounds
    """
    if cache.grid is not None:
        return cache.grid

    if provider is None:
        provider = get_data_provider()

    coords = provider.get_coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_spacing = float(np.min(np.diff(x_unique))) if len(x_unique) > 1 else 1.0
    y_spacing = float(np.min(np.diff(y_unique))) if len(y_unique) > 1 else 1.0

    grid_map: dict[tuple[int, int], list[int]] = {}
    for idx, (xi, yi) in enumerate(zip(x, y, strict=True)):
        grid_x = int(np.round(float(xi) / x_spacing))
        grid_y = int(np.round(float(yi) / y_spacing))
        grid_map.setdefault((grid_x, grid_y), []).append(idx)

    grid_cache: dict[str, Any] = {
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
    reference_date: str,
    *,
    radius: float = 10.0,
    method: Literal["hybrid", "kdtree", "grid"] = "hybrid",
    provider: DataProvider | None = None,
) -> dict[str, Any] | None:
    """Find the nearest DIC node to (x, y) for a given reference date.

    The reference date is the FINAL image date used for the DIC computation.

    Args:
        x: Query x coordinate.
        y: Query y coordinate.
        reference_date: Reference date in YYYYMMDD (final image date).
        radius: Maximum allowed distance for a match.
        method: Search method: "grid", "kdtree", or "hybrid".
        provider: Optional data provider. If None, uses the default provider.

    Returns:
        Node data dict, or None if not found/available.
    """
    logger.debug(
        "nearest_node: query (%s,%s) date=%s method=%s radius=%s",
        x,
        y,
        reference_date,
        method,
        radius,
    )

    if provider is None:
        provider = get_data_provider()

    payload = provider.get_dic_data(reference_date)
    if not payload:
        logger.debug(
            "nearest_node: reference_date %s not found/available", reference_date
        )
        return None

    if method == "kdtree":
        tree, coords = build_kdtree(provider)
        return _nearest_node_kdtree(tree, coords, payload, x, y, radius)

    if method == "grid":
        grid_cache = build_grid_index(provider)
        return _nearest_node_grid(grid_cache, payload, x, y, radius)

    if method == "hybrid":
        if radius <= 100:
            grid_cache = build_grid_index(provider)
            return _nearest_node_grid(grid_cache, payload, x, y, radius)
        tree, coords = build_kdtree(provider)
        return _nearest_node_kdtree(tree, coords, payload, x, y, radius)

    raise ValueError(f"Unknown method '{method}' - expected 'hybrid'|'kdtree'|'grid'")


def _nearest_node_kdtree(
    tree: cKDTree,
    coords: np.ndarray,
    payload: dict[str, Any],
    x: float,
    y: float,
    radius: float = 50.0,
) -> dict[str, Any] | None:
    """Find nearest node using a KDTree for a single payload.

    Args:
        tree: Pre-built KDTree for the coordinate set.
        coords: `(N, 2)` array of coordinates matching the payload arrays.
        payload: DIC payload for the selected record/date.
        x: Query x coordinate.
        y: Query y coordinate.
        radius: Search radius (distance upper bound).

    Returns:
        Node data dict or None if no node is found within radius.
    """
    logger.debug("nearest_node_kdtree: query at (%s, %s) radius=%s", x, y, radius)

    dist, idx = tree.query([x, y], k=1, distance_upper_bound=radius)
    if not np.isfinite(dist) or idx >= len(coords):
        logger.debug("nearest_node_kdtree: no node within radius %s", radius)
        return None

    return _extract_node_data(payload, int(idx))


def _nearest_node_grid(
    grid_cache: dict[str, Any],
    payload: dict[str, Any],
    x: float,
    y: float,
    radius: float = 50.0,
) -> dict[str, Any] | None:
    """Find nearest node using a grid index for a single payload.

    Args:
        grid_cache: Output of `build_grid_index()`.
        payload: DIC payload for the selected record/date.
        x: Query x coordinate.
        y: Query y coordinate.
        radius: Search radius.

    Returns:
        Node data dict or None if no node is found within radius.
    """
    logger.debug("nearest_node_grid: query at (%s, %s) radius=%s", x, y, radius)

    grid_map: dict[tuple[int, int], list[int]] = grid_cache["map"]
    x_spacing: float = float(grid_cache["x_spacing"])
    y_spacing: float = float(grid_cache["y_spacing"])
    coords: np.ndarray = grid_cache["coords"]

    grid_x = int(np.round(x / x_spacing))
    grid_y = int(np.round(y / y_spacing))

    max_search_cells = int(np.ceil(radius / min(x_spacing, y_spacing))) + 1

    best_dist = float("inf")
    best_idx: int | None = None

    for search_radius in range(max_search_cells + 1):
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                # scan only the "ring" for efficiency
                if abs(dx) < search_radius and abs(dy) < search_radius:
                    continue

                key = (grid_x + dx, grid_y + dy)
                ids = grid_map.get(key)
                if not ids:
                    continue

                for idx in ids:
                    node_x, node_y = coords[idx]
                    dist = float(np.hypot(x - float(node_x), y - float(node_y)))
                    if dist <= radius and dist < best_dist:
                        best_dist = dist
                        best_idx = int(idx)

        if best_idx is not None and search_radius > 0:
            break

    if best_idx is None:
        logger.debug("nearest_node_grid: no node found within radius %s", radius)
        return None

    return _extract_node_data(payload, best_idx)


def _extract_node_data(payload: dict[str, Any], idx: int) -> dict[str, Any]:
    """Extract all payload values at a specific point index.

    Args:
        payload: DIC payload with array-like values.
        idx: Point index.

    Returns:
        A dict with scalar values for the selected index. Scalars are converted
        to Python `float`/`int` when possible.
    """
    result: dict[str, Any] = {}
    for k, v in payload.items():
        if k in ("dt_days", "dt_hours", "master_date", "slave_date"):
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
