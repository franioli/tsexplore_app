"""Centralized data cache manager.

This cache stores one DIC result per record (typically one file read by the loader). Multiple records can share the same reference date, but different dt (e.g., X-1, X-3, X-5 intervals).

Design:
- Primary storage is keyed by record_id:
    - dic_records[record_id] -> payload dict (arrays + precomputed velocity)
    - dic_meta[record_id] -> DicMeta (metadata: initial/final dates, dt, etc)
- Indexes:
    - records_by_reference[YYYYMMDD] -> list of record_ids (final image date)
- Derived structures (built lazily):
    - kdtree: spatial index for fast nearest-node queries
    - images: cached background images (data URLs + raw bytes)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore


@dataclass(frozen=True)
class DicMeta:
    """Metadata describing a single DIC result (a single loaded file).

    Terminology:
        - initial_date: date of the initial image used in the DIC computation.
        - final_date: date of the final image used in the DIC computation.
        - reference_date: alias for final_date (the day the result is referenced to).

    Attributes:
        record_id: Unique identifier for this cached record. Assigned by DataCache.
        initial_date: Initial date (initial image) of the DIC analysis.
        final_date: Final date (final image) of the DIC analysis (aka reference date).
        dt_hours: Time span between initial and final in hours.
        dt_days: Time span between initial and final in days (integer).
    """

    record_id: int
    initial_date: datetime
    final_date: datetime
    dt_hours: int
    dt_days: int

    @property
    def reference_date(self) -> datetime:
        """Alias for final_date.

        Returns:
            The final date of the DIC pair (the reference date).
        """
        return self.final_date


class DataCache:
    """In-memory cache for DIC results, spatial indexes, and background images."""

    def __init__(self) -> None:
        """Initialize an empty cache."""
        # record storage
        self.dic_records: dict[int, dict[str, Any]] = {}
        self.dic_meta: dict[int, DicMeta] = {}

        # index: reference day (YYYYMMDD) -> list[record_id]
        # Reference date is the FINAL image date used for the DIC computation.
        self.records_by_reference: dict[str, list[int]] = {}

        # derived structures (built lazily)
        self.kdtree: tuple[cKDTree, np.ndarray] | None = None  # (tree, coords)
        self.images: dict[str, dict[str, Any]] = {}

        # loading/progress metadata for DB range loads
        self.load_in_progress: bool = False
        self.load_total: int = 0
        self.load_done: int = 0
        self.load_start_date: str | None = None
        self.load_end_date: str | None = None
        self.load_error: str | None = None

        # Last record id used (for assigning new record ids)
        self._last_record_id: int = 0

    def __repr__(self) -> str:
        """Return a short representation useful for logs/debug."""
        return f"<DataCache records={self.num_records}>"

    @property
    def num_records(self) -> int:
        """Return the number of stored DIC records."""
        return len(self.dic_records)

    def is_loaded(self) -> bool:
        """Return True if at least one record is loaded."""
        return bool(self.dic_records)

    def has_kdtree(self) -> bool:
        """Return True if the KDTree is already built and cached."""
        return self.kdtree is not None

    def get_available_dates(self) -> list[str]:
        """Return loaded reference dates as YYYYMMDD strings.

        The reference date corresponds to the final image date of the DIC pair.
        """
        return sorted(self.records_by_reference.keys())

    def get_dt_days_for_date(self, reference_yyyymmdd: str) -> list[int]:
        """Return available dt_days values for a given reference day.

        Args:
            reference_yyyymmdd: Reference date in YYYYMMDD format (final image date).

        Returns:
            Sorted unique list of dt_days available for that reference day.
        """
        metas = list(self._iter_metas_for_reference(reference_yyyymmdd))
        return sorted({m.dt_days for m in metas})

    def store_dic_record(
        self, *, meta: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        """Store one DIC record and index it by reference date (final date).

        Args:
            meta: Record metadata (initial/final/reference + dt).
            payload: Record payload (arrays + derived fields).
        """
        # Build DicMeta object
        id = self._last_record_id + 1
        dic_meta = DicMeta(
            record_id=id,
            initial_date=meta["initial_date"],
            final_date=meta["final_date"],
            dt_hours=meta["dt_hours"],
            dt_days=meta["dt_days"],
        )

        # Store record
        self.dic_meta[id] = dic_meta
        self.dic_records[id] = payload
        reference_key = dic_meta.reference_date.strftime("%Y%m%d")
        self.records_by_reference.setdefault(reference_key, []).append(id)

        # Update last record id
        self._last_record_id = id

    def get_dic_record(self, record_id: int) -> tuple[DicMeta, dict[str, Any]] | None:
        """Get a cached record by record_id.

        Args:
            record_id: The record identifier.

        Returns:
            Tuple (meta, payload) if present, otherwise None.
        """
        meta = self.dic_meta.get(record_id)
        rec = self.dic_records.get(record_id)
        if meta is None or rec is None:
            return None
        return meta, rec

    def pick_record_id_for_date(
        self,
        reference_yyyymmdd: str,
        *,
        dt_days: int | None = None,
        prefer: int | None = None,
        prefer_tolerance: int | None = None,
    ) -> int | None:
        """Pick one record_id for a reference day (final-image date).

        Args:
            reference_yyyymmdd: Reference date in YYYYMMDD format.
            dt_days: Optional exact dt_days to match.
            prefer: Optional preferred dt_days for closest match.
            prefer_tolerance: Optional maximum allowed deviation from prefer.
        """
        metas = list(self._iter_metas_for_reference(reference_yyyymmdd))
        if not metas:
            return None

        if dt_days is not None:
            target = int(dt_days)
            for m in metas:
                if m.dt_days == target:
                    return m.record_id
            return None

        if prefer is not None:
            pref = int(prefer)
            best = sorted(metas, key=lambda m: (abs(m.dt_days - pref), m.record_id))[0]
            if prefer_tolerance is not None:
                tol = max(0, int(prefer_tolerance))
                if abs(best.dt_days - pref) > tol:
                    return None
            return best.record_id

        # default: smallest dt_days, then smallest id
        best = sorted(metas, key=lambda m: (m.dt_days, m.record_id))[0]
        return best.record_id

    def find_record_id(
        self,
        *,
        reference_yyyymmdd: str,
        initial_yyyymmdd: str | None = None,
        dt_days: int | None = None,
        prefer_dt_days: int | None = None,
        prefer_dt_tolerance: int | None = None,
    ) -> int | None:
        """Find a record id using metadata constraints.

        Args:
            reference_yyyymmdd: Reference date in YYYYMMDD format (final image date).
            initial_yyyymmdd: Optional initial date in YYYYMMDD format.
            dt_days: Optional exact dt_days.
            prefer_dt_days: Optional preferred dt_days for closest match.
            prefer_dt_tolerance: Optional tolerance (days) for the closest-match selection.
        """
        if initial_yyyymmdd is not None:
            target_initial = initial_yyyymmdd
            target_dt = int(dt_days) if dt_days is not None else None

            for m in self._iter_metas_for_reference(reference_yyyymmdd):
                if m.initial_date.strftime("%Y%m%d") != target_initial:
                    continue
                if target_dt is not None and m.dt_days != target_dt:
                    continue
                return m.record_id
            return None

        return self.pick_record_id_for_date(
            reference_yyyymmdd,
            dt_days=dt_days,
            prefer=prefer_dt_days,
            prefer_tolerance=prefer_dt_tolerance,
        )

    def clear(self) -> None:
        """Clear all cached records, indexes, and derived structures."""
        self.dic_records.clear()
        self.dic_meta.clear()
        self.records_by_reference.clear()

        self.kdtree = None
        self.images.clear()

        self.load_in_progress = False
        self.load_total = 0
        self.load_done = 0
        self.load_start_date = None
        self.load_end_date = None
        self.load_error = None

    def _iter_metas_for_reference(self, reference_yyyymmdd: str) -> Iterable[DicMeta]:
        """Yield DicMeta entries for a given reference date.

        Args:
            reference_yyyymmdd: Reference date in YYYYMMDD format (final image date).

        Yields:
            DicMeta items for records associated with the reference date.
        """
        ids = self.records_by_reference.get(reference_yyyymmdd, [])
        for rid in ids:
            m = self.dic_meta.get(rid)
            if m is not None:
                yield m

    # ---------------------------------------------------------------------
    # Image caching
    # ---------------------------------------------------------------------
    def get_image(
        self, path: str
    ) -> tuple[str | None, tuple[int, int] | None, bytes | None]:
        """Get a cached image by path.

        Args:
            path: Image path used as cache key.

        Returns:
            Tuple (data_url, size, raw_bytes). Each entry is None if not cached.
        """
        if path in self.images:
            img = self.images[path]
            return img["data_url"], img["size"], img["bytes"]
        return (None, None, None)

    def store_image(
        self, path: str, data_url: str, size: tuple[int, int], raw_bytes: bytes
    ) -> None:
        """Store an image in the cache.

        Args:
            path: Cache key (usually the file path).
            data_url: Base64 data URL used by Plotly layout images.
            size: Image size in pixels as (width, height).
            raw_bytes: Original encoded bytes (e.g., PNG/JPEG).
        """
        self.images[path] = {"data_url": data_url, "size": size, "bytes": raw_bytes}


class AppState:
    """Application state tracker."""

    def __init__(self) -> None:
        """Initialize the application state."""
        self._ready = False

    def is_ready(self) -> bool:
        """Return True if the application is ready to serve requests."""
        return self._ready

    def mark_ready(self) -> None:
        """Mark the application as ready."""
        self._ready = True


cache = DataCache()
