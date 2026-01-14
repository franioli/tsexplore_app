"""Spatial indexing and nearest-neighbor utilities.

This module provides:
- A cached KDTree for fast nearest-node lookup.
- A high-level `nearest_node()` helper that fetches the DIC payload for a given
  slave date via the active DataProvider, and returns values at the nearest node.
"""

import logging
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from ..cache import cache
from ..dataloaders.data_provider import DataProvider, get_data_provider

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


def nearest_node(
    x: float,
    y: float,
    reference_date: str,
    *,
    radius: float = 10.0,
    provider: DataProvider | None = None,
) -> dict[str, Any] | None:
    """Find the nearest DIC node to (x, y) for a given reference date.

    The reference date is the FINAL image date used for the DIC computation.

    Args:
        x: Query x coordinate.
        y: Query y coordinate.
        reference_date: Reference date in YYYYMMDD (final image date).
        radius: Maximum allowed distance for a match.
        provider: Optional data provider. If None, uses the default provider.

    Returns:
        Node data dict, or None if not found/available.
    """
    logger.debug(f"nearest_node: query ({x},{y}) date={reference_date} radius={radius}")

    if provider is None:
        provider = get_data_provider()

    payload = provider.get_dic_data(reference_date)
    if not payload:
        logger.debug(
            "nearest_node: reference_date %s not found/available", reference_date
        )
        return None

    # Build or fetch KDTree
    tree, coords = build_kdtree(provider)

    # Find nearest node
    logger.debug(f"nearest_node_kdtree: query at ({x}, {y}) radius={radius}")
    dist, node_idx = tree.query([x, y], k=1, distance_upper_bound=radius)
    if not np.isfinite(dist) or node_idx >= len(coords):
        logger.debug(f"nearest_node_kdtree: no node within radius {radius}")
        return None

    if node_idx is None:
        return None

    # Extract node data at found index
    result: dict[str, Any] = {}
    result["node_id"] = int(node_idx)
    key_to_extract = ["x", "y", "dx", "dy", "disp_mag", "u", "v", "V", "ensemble_mad"]
    for key in key_to_extract:
        if key in payload:
            result[key] = payload[key][node_idx]

    return result
