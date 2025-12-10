"""Data provider interface and factory."""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import Settings, get_settings


@runtime_checkable
class DataProvider(Protocol):
    """Interface for DIC data providers (file-based or database)."""

    def get_available_dates(self) -> list[str]:
        """Return list of available dates in YYYYMMDD format."""
        ...

    def get_dic_data(self, date: str) -> dict[str, np.ndarray] | None:
        """
        Get DIC data for a specific date.

        Returns:
            Dictionary with keys: x, y, dx, dy, disp_mag, u, v, V,
            ensamble_mad, dt_hours, dt_days
        """
        ...

    def preload_all(self) -> dict[str, dict]:
        """Preload all available DIC data into memory."""
        ...

    def get_coordinates(self) -> np.ndarray:
        """Get reference coordinate array (N, 2) for spatial indexing."""
        ...

    def preload_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, dict]:
        """
        Preload DIC data for the desired date range.
        progress_callback(done, total) is called for progress updates.
        """
        ...


def get_data_provider(settings: Settings | None = None) -> DataProvider:
    """
    Factory function to get the configured data provider.

    Returns file or database provider based on settings.
    """

    if settings is None:
        settings = get_settings()

    if settings.use_database:
        from .db_loader import DatabaseDataProvider

        return DatabaseDataProvider()
    else:
        from .file_loader import FileDataProvider

        return FileDataProvider()
