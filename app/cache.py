"""Centralized data cache manager."""

from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore


###=== Data Cache Manager ===###
class DataCache:
    """Centralized data cache manager."""

    def __init__(self):
        self.all_data: dict[str, dict] | None = None
        self.kdtree: tuple[cKDTree, np.ndarray] | None = None  # (tree, coords)
        self.grid: dict[str, Any] | None = None
        self.images: dict[str, dict[str, Any]] = {}

        # loading/progress metadata for DB range loads
        self.load_in_progress: bool = False
        self.load_total: int = 0
        self.load_done: int = 0
        self.load_start_date: str | None = None
        self.load_end_date: str | None = None
        self.load_error: str | None = None

    @property
    def num_dates(self) -> int | None:
        """Get number of loaded dates."""
        if self.all_data is None:
            return None
        return len(self.all_data)

    def get_image(
        self, path: str
    ) -> tuple[str | None, tuple[int, int] | None, bytes | None]:
        """Return cached image (data_url, size, raw_bytes) if present."""
        if path in self.images:
            img = self.images[path]
            return img["data_url"], img["size"], img["bytes"]
        return (None, None, None)

    def store_image(
        self, path: str, data_url: str, size: tuple[int, int], raw_bytes: bytes
    ) -> None:
        """Store image metadata and raw bytes in cache."""
        self.images[path] = {
            "data_url": data_url,
            "size": size,
            "bytes": raw_bytes,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        self.all_data = None
        self.kdtree = None
        self.grid = None
        self.images = {}
        # reset loading metadata
        self.load_in_progress = False
        self.load_total = 0
        self.load_done = 0
        self.load_start_date = None
        self.load_end_date = None
        self.load_error = None

    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self.all_data is not None

    def has_kdtree(self) -> bool:
        """Check if spatial indices are built."""
        return self.kdtree is not None

    def has_grid(self) -> bool:
        """Check if grid index is built."""
        return self.grid is not None


class AppState:
    """Application state tracker."""

    def __init__(self):
        self._ready = False

    def is_ready(self) -> bool:
        """Check if app is ready to serve requests."""
        return self._ready

    def mark_ready(self):
        """Mark application as ready."""
        self._ready = True


# Global cache instance
cache = DataCache()
