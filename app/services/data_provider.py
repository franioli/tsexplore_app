"""Data provider interface and factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import Settings, get_settings


def get_data_provider(settings: Settings | None = None) -> DataProvider:
    """
    Factory function to get the configured data provider.

    Returns file or database provider based on settings.
    """

    if settings is None:
        settings = get_settings()

    if settings.use_database:
        from .db_loader import DatabaseDataProvider

        return DatabaseDataProvider(settings=settings)
    else:
        from .file_loader import FileDataProvider

        return FileDataProvider(settings=settings)


@runtime_checkable
class DataProvider(Protocol):
    """Interface for DIC data providers (file-based or database)."""

    def get_available_dates(self) -> list[str]:
        """Return all *available* reference dates (YYYY-MM-DD) WITHOUT loading full data.

        This must be cheap:
        - file backend: scan filenames and parse dates
        - db backend: query distinct dates / metadata table
        """
        raise NotImplementedError

    def get_dic_data(
        self,
        reference_date: str,
        *,
        dt_days: int | None = None,
        initial_date: str | None = None,
        prefer_dt_days: int | None = None,
        prefer_dt_tolerance: int | None = None,
    ) -> dict[str, Any] | None:
        """Return DIC payload for a reference date with optional interval selection."""
        ...

    def load_all(self) -> int:
        """Load all available DIC data into memory."""
        ...

    def load_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """
        Load DIC data for the desired date range.
        progress_callback(done, total) is called for progress updates.
        """
        ...

    def get_coordinates(self) -> np.ndarray:
        """Get reference coordinate array (N, 2) for spatial indexing."""
        ...

    def extract_node_timeseries(
        self,
        node_x: float,
        node_y: float,
        *,
        dt_days: int | None = None,
        dt_hours_tolerance: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Extract a time series for the nearest mesh node at (node_x, node_y).

        This returns a single timeseries (not grouped). If ``dt_days`` is given,
        only records whose actual time difference (final - initial) is within
        ``dt_days_tolerance`` days of ``dt_days`` are included. Tolerances are
        expressed in fractional days (e.g. 0.5 = 12 hours).

        Args:
            node_x: X coordinate of the requested node.
            node_y: Y coordinate of the requested node.
            dt_days: If provided, filter records to those with dt close to this value (days).
            dt_days_tolerance: Allowed absolute tolerance around ``dt_days`` (in days).

        Returns:
            A dict with numpy arrays for keys:
              'reference_dates', 'initial_dates', 'final_dates', 'dt_days',
              'dx', 'dy', 'disp_mag', 'u', 'v', 'V', 'ensemble_mad'
            If no matching records are found an empty dict is returned.
        """
        ...
