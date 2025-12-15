"""File-based DIC data provider.

This module implements a file-backed DataProvider that loads *one cache record per file*.

Key points
- A single "slave day" (YYYYMMDD) may have multiple DIC results (different dt / master date).
- Records are stored in the global cache with metadata (master_date, slave_date, dt).
- Selection for a given slave day is done via cache.find_record_id(...):
  - exact dt_days, or
  - (master_date, dt_days), or
  - prefer_dt_days (closest), otherwise defaults to smallest dt_days.
"""

import logging
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from ..cache import cache
from ..config import get_settings
from .data_provider import DataProvider

logger = logging.getLogger(__name__)


class FileDataProvider(DataProvider):
    """Load DIC data from text files on disk."""

    def __init__(self):
        """Initialize the file-backed provider."""
        self.settings = get_settings()
        self._data_cache = None

    def get_available_dates(self) -> list[str]:
        """Return available (loaded) slave dates.

        Returns:
            A sorted list of slave dates as strings.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if not cache.is_loaded():
            self.load_all()
        return cache.get_available_dates()

    def get_dic_data(
        self,
        reference_date: str,
        *,
        dt_days: int | None = None,
        initial_date: str | None = None,
        prefer_dt_days: int | None = None,
        prefer_dt_tolerance: int | None = None,
    ) -> dict | None:
        """Get the DIC payload for a given reference day (final image date).

        Args:
            reference_date: Reference date in YYYYMMDD (final image date).
            dt_days: If provided, select the record with this time interval (days).
            initial_date: If provided, select by initial date (YYYYMMDD). Can be combined
                with dt_days for an exact match.
            prefer_dt_days: If provided, pick the record whose dt_days is closest.
            prefer_dt_tolerance: Optional tolerance (days) for closest match.

        Returns:
            The payload dict for the selected record, or None if no matching record exists.
        """
        if not cache.is_loaded():
            self.load_all()

        rid = cache.find_record_id(
            reference_yyyymmdd=reference_date,
            initial_yyyymmdd=initial_date,
            dt_days=dt_days,
            prefer_dt_days=prefer_dt_days,
            prefer_dt_tolerance=prefer_dt_tolerance,
        )
        if rid is None:
            return None

        rec = cache.get_dic_record(rid)
        if rec is None:
            return None
        _meta, payload = rec
        return payload

    def load_all(self) -> int:
        """Load all DIC files from disk into the global cache.

        Returns:
            The number of loaded records (one record per file).
        """
        if cache.is_loaded():
            self._loaded = True
            return cache.num_records

        s = self.settings
        n = _load_all_from_disk_into_cache(
            data_dir=s.data_dir,
            file_pattern=s.file_pattern,
            filename_pattern=s.filename_pattern,
            date_format=s.date_format,
            invert_y=s.invert_y,
            dt_days_preferred=getattr(s, "dt_days_preferred", None),
        )
        self._loaded = True
        return n

    def load_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Load only the dates in the requested range (files).

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            progress_callback: Optional callback receiving ``(done, total)``.

        Returns:
            Number of loaded records.
        """
        raise NotImplementedError("FileDataProvider does not support load_range()")

    def get_coordinates(self) -> np.ndarray:
        """Get coordinate array from the first loaded record.

        Returns:
            (N, 2) array with columns [x, y].

        Raises:
            ValueError: If no data is available.
        """
        if not cache.is_loaded():
            self.load_all()

        if cache.num_records == 0:
            raise ValueError("No data available")

        # Pick the first record_id deterministically
        first_id = sorted(cache.dic_records.keys())[0]
        rec = cache.get_dic_record(first_id)
        if rec is None:
            raise ValueError("No data available")

        _meta, payload = rec
        return np.column_stack([payload["x"], payload["y"]])


def _parse_filename(
    file_path: Path, *, filename_pattern: str, date_format: str
) -> tuple[datetime, datetime]:
    """Parse initial/final dates from a DIC filename.

    The reference date used by the app is the FINAL image date.

    Args:
        file_path: Path to the file.
        filename_pattern: Regex pattern that extracts initial and final date strings as the first two capture groups.
        date_format: datetime.strptime format for the captured date strings.

    Returns:
        Tuple (final_date, initial_date).

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    pattern = re.compile(filename_pattern)
    match = pattern.search(file_path.name)
    if not match or len(match.groups()) < 2:
        raise ValueError(f"Invalid filename: {file_path.name}")

    final_str = match.group(1)
    initial_str = match.group(2)
    final_date = datetime.strptime(final_str, date_format)
    initial_date = datetime.strptime(initial_str, date_format)

    return final_date, initial_date


def _read_txt(file_path: Path, invert_y: bool) -> dict[str, np.ndarray]:
    """Read a DIC text file.

    Args:
        file_path: Path to the ``.txt`` file.
        invert_y: Whether to invert the y-axis (and dy) sign.

    Returns:
        A dict containing arrays: ``x``, ``y``, ``dx``, ``dy``, ``magnitude``,
        ``ensamble_mad``.

    Raises:
        ValueError: If the file is empty.
    """
    data = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=0,
        dtype=np.float32,
        ndmin=2,
    )
    if data.size == 0:
        raise ValueError("Empty data")

    x = data[:, 0]
    y = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]

    if invert_y:
        y = -y
        dy = -dy

    magnitude = np.hypot(dx, dy)
    ensamble_mad = data[:, 4] if data.shape[1] > 4 else np.zeros_like(magnitude)

    return {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
        "ensamble_mad": ensamble_mad,
    }


def _read_h5(file_path: Path, invert_y: bool) -> dict[str, np.ndarray]:
    """Read a DIC HDF5 file.

    Args:
        file_path: Path to the ``.h5`` file.
        invert_y: Whether to invert the y-axis (and dy) sign.

    Returns:
        A dict containing arrays: ``x``, ``y``, ``dx``, ``dy``, ``magnitude``,
        ``ensamble_mad``.
    """
    import h5py

    with h5py.File(file_path, "r") as f:
        x = f["x"][:]
        y = f["y"][:]
        dx = f["dx"][:]
        dy = f["dy"][:]

        if invert_y:
            y = -y
            dy = -dy

        magnitude = np.hypot(dx, dy)
        ensamble_mad = (
            f["ensamble_mad"][:] if "ensamble_mad" in f else np.zeros_like(magnitude)
        )

    return {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
        "ensamble_mad": ensamble_mad,
    }


def _read_dic_file(file_path: Path, *, invert_y: bool) -> dict[str, np.ndarray]:
    """Read a DIC file based on its extension.

    Args:
        file_path: Path to the DIC file.
        invert_y: Whether to invert the y-axis (and dy) sign.

    Returns:
        A dict containing arrays: ``x``, ``y``, ``dx``, ``dy``, ``magnitude``,
        ``ensamble_mad``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".h5":
        return _read_h5(file_path, invert_y)
    if suffix == ".txt":
        return _read_txt(file_path, invert_y)

    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _load_all_from_disk_into_cache(
    *,
    data_dir: str | Path,
    file_pattern: str,
    filename_pattern: str,
    date_format: str,
    invert_y: bool,
    dt_days_preferred: int | None,
    dt_days_tolerance: int = 1,
) -> int:
    """Load all DIC files from disk into the global cache (one record per file).

    Args:
        data_dir: Root directory containing DIC files.
        file_pattern: Glob pattern used to list files (e.g. ``"*.txt"``).
        filename_pattern: Regex pattern extracting slave/master date strings.
        date_format: Format string for parsing the extracted date strings.
        invert_y: Whether to invert the y-axis (and dy) sign.
        dt_days_preferred: If set (> 0), keep only records within ``dt_days_tolerance``
            from the preferred dt. If ``None``, all records are kept.
        dt_days_tolerance: Allowed absolute deviation from ``dt_days_preferred`` in days.

    Returns:
        Number of loaded records stored into the global cache.
    """
    logger.info("Loading DIC files from disk..")
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_pattern))

    loaded = 0
    for f in files:
        try:
            final_date, initial_date = _parse_filename(
                f, filename_pattern=filename_pattern, date_format=date_format
            )

            # Compute time difference
            dt_hours = int(abs((final_date - initial_date).total_seconds()) / 3600)
            dt_days = int(dt_hours / 24)

            # Optional filtering by preferred dt
            if dt_days_preferred is not None and int(dt_days_preferred) > 0:
                if abs(dt_days - int(dt_days_preferred)) > max(
                    0, int(dt_days_tolerance)
                ):
                    continue

            dic_data = _read_dic_file(f, invert_y=invert_y)
            dx = dic_data["dx"]
            dy = dic_data["dy"]
            disp_mag = dic_data["magnitude"]

            # Compute velocity (px/day) if dt > 24h, otherwise keep displacement as-is
            if dt_hours > 24:
                scale = 24.0 / dt_hours
                u_vel = dx * scale
                v_vel = dy * scale
                V_vel = disp_mag * scale
            else:
                u_vel = dx
                v_vel = dy
                V_vel = disp_mag

            meta = {
                "initial_date": initial_date,
                "final_date": final_date,
                "reference_date": final_date,
                "dt_hours": dt_hours,
                "dt_days": dt_days,
            }
            payload = {
                "x": dic_data["x"],
                "y": dic_data["y"],
                "dx": dx,
                "dy": dy,
                "disp_mag": disp_mag,
                "u": u_vel,
                "v": v_vel,
                "V": V_vel,
                "ensamble_mad": dic_data["ensamble_mad"],
                "dt_hours": dt_hours,
                "dt_days": dt_days,
                # keep dates in payload too (handy for API responses/metadata)
                "initial_date": initial_date.strftime("%Y%m%d"),
                "final_date": final_date.strftime("%Y%m%d"),
                "reference_date": final_date.strftime("%Y%m%d"),
            }

            cache.store_dic_record(meta=meta, payload=payload)
            loaded += 1

        except Exception as e:
            logger.debug(f"Skipping {f.name}: {e}")
            continue

    logger.info(f"Loaded {loaded} DIC files from disk (records={cache.num_records})")
    return loaded
