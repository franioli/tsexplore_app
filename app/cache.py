"""Centralized data cache manager."""

from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore


###=== Data Cache Manager ===###
class DataCache:
    """Centralized data cache manager."""

    def __init__(self):
        self._all_data: dict[str, dict] | None = None
        self._kdtree: tuple[cKDTree, np.ndarray] | None = None
        self._grid: dict[str, Any] | None = None
        self._images: dict[str, dict[str, Any]] = {}

    @property
    def all_data(self) -> dict[str, dict] | None:
        return self._all_data

    @all_data.setter
    def all_data(self, value: dict[str, dict]) -> None:
        self._all_data = value

    @property
    def kdtree(self) -> tuple[cKDTree, np.ndarray] | None:
        return self._kdtree

    @kdtree.setter
    def kdtree(self, value: tuple[cKDTree, np.ndarray]) -> None:
        self._kdtree = value

    @property
    def grid(self) -> dict[str, Any] | None:
        return self._grid

    @grid.setter
    def grid(self, value: dict[str, Any]) -> None:
        self._grid = value

    @property
    def num_dates(self) -> int | None:
        """Get number of loaded dates."""
        if self._all_data is None:
            return None
        return len(self._all_data)

    @property
    def images(self) -> dict[str, dict[str, Any]]:
        return self._images

    def get_image(
        self, path: str
    ) -> tuple[str | None, tuple[int, int] | None, bytes | None]:
        """Return cached image (data_url, size, raw_bytes) if present."""
        if path in self._images:
            img = self._images[path]
            return img["data_url"], img["size"], img["bytes"]
        return (None, None, None)

    def store_image(
        self, path: str, data_url: str, size: tuple[int, int], raw_bytes: bytes
    ) -> None:
        """Store image metadata and raw bytes in cache."""
        self._images[path] = {
            "data_url": data_url,
            "size": size,
            "bytes": raw_bytes,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        self._all_data = None
        self._kdtree = None
        self._grid = None
        self._images = {}

    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._all_data is not None

    def has_kdtree(self) -> bool:
        """Check if spatial indices are built."""
        return self._kdtree is not None or self._grid is not None


# Global cache instance
cache = DataCache()
