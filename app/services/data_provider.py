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

        return DatabaseDataProvider()
    else:
        from .file_loader import FileDataProvider

        return FileDataProvider()


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
        group_by_dt: bool = True,
        delta_days: list[int] | None = None,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Extract a node time series across all available slave dates, optionally grouped by dt.

        Returns a dict mapping group_dt -> timeseries dict with numpy arrays. If group_by_dt is False
        a single group with key 0 is returned containing all records.

        If delta_days is provided it is used as the list of dt bucket centers; each record's dt_days
        is rounded to the nearest provided delta_days value.
        """
        ...
