"""File-based DIC data provider."""

import logging
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from ..cache import cache
from ..config import get_settings
from .provider import DataProvider

logger = logging.getLogger(__name__)


class FileDataProvider(DataProvider):
    """Load DIC data from text files on disk."""

    def __init__(self):
        self.settings = get_settings()
        self._data_cache = None

    def get_available_dates(self) -> list[str]:
        """Scan directory for available dates."""
        if self._data_cache is None:
            raise ValueError("Data not loaded yet. Load data first.")
        return sorted(self._data_cache.keys())

    def get_dic_data(self, date: str) -> dict | None:
        """Get data for a specific date."""
        data = self._data_cache or self.load_all()
        if not data:
            return None

        return data.get(date)

    def load_all(self) -> dict[str, dict]:
        """load all DIC files from disk."""
        # Use global cache if available
        if cache.all_data is not None:
            return cache.all_data

        # Use instance cache if available
        if self._data_cache is not None:
            return self._data_cache

        s = self.settings
        all_data = _load_all_from_disk(
            data_dir=s.data_dir,
            file_pattern=s.file_pattern,
            filename_pattern=s.filename_pattern,
            date_format=s.date_format,
            invert_y=s.invert_y,
            dt_days_preferred=getattr(s, "dt_days_preferred", None),
        )

        cache.all_data = all_data
        self._data_cache = all_data
        return all_data

    def load_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, dict]:
        """Load only the dates in the requested range (files)."""
        all_data = self.load_all()
        filtered = _filter_data_by_date_range(
            all_data=all_data,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback,
        )

        # cache the filtered set (keeps current behavior)
        cache.all_data = filtered
        self._data_cache = filtered
        return filtered

    def get_coordinates(self) -> np.ndarray:
        """Get coordinate array from first available date."""

        # Ensure data is loaded
        all_data = self._data_cache or self.load_all()
        if not all_data:
            raise ValueError("No data available")

        first_date = next(iter(all_data.keys()))
        first_data = all_data[first_date]
        return np.column_stack([first_data["x"], first_data["y"]])


def _filter_data_by_date_range(
    *,
    all_data: dict[str, dict],
    start_date: str,
    end_date: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Filter a date-keyed dict by [start_date, end_date] (inclusive)."""

    def _to_dateobj(datestring: str):
        """Parse YYYYMMDD or YYYY-MM-DD into a date."""
        if "-" in datestring:
            return datetime.strptime(datestring, "%Y-%m-%d").date()
        return datetime.strptime(datestring, "%Y%m%d").date()

    start_obj = _to_dateobj(start_date)
    end_obj = _to_dateobj(end_date)

    dates = sorted(all_data.keys())
    selected_dates = [
        d
        for d in dates
        if start_obj <= datetime.strptime(d, "%Y%m%d").date() <= end_obj
    ]

    total = len(selected_dates)
    done = 0

    out: dict[str, dict] = {}
    for d in selected_dates:
        out[d] = all_data[d]
        done += 1
        if progress_callback:
            progress_callback(done, total)

    return out


def _parse_filename(
    file_path: Path, *, filename_pattern: str, date_format: str
) -> tuple[datetime, datetime]:
    """Parse slave/master dates from filename."""
    pattern = re.compile(filename_pattern)
    match = pattern.search(file_path.name)
    if not match or len(match.groups()) < 2:
        raise ValueError(f"Invalid filename: {file_path.name}")

    slave_str = match.group(1)
    master_str = match.group(2)
    slave_date = datetime.strptime(slave_str, date_format)
    master_date = datetime.strptime(master_str, date_format)
    return slave_date, master_date


def _read_txt(file_path: Path, invert_y: bool) -> dict[str, np.ndarray]:
    """Read a DIC text file."""
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
    """Read a DIC HDF5 file."""
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
    """Read a DIC file based on its extension."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".h5":
        return _read_h5(file_path, invert_y)
    if suffix == ".txt":
        return _read_txt(file_path, invert_y)

    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _load_all_from_disk(
    *,
    data_dir: str | Path,
    file_pattern: str,
    filename_pattern: str,
    date_format: str,
    invert_y: bool,
    dt_days_preferred: int | None,
    dt_days_tolerance: int = 1,
) -> dict[str, dict]:
    """Load all DIC files from disk into a date-keyed dict."""
    logger.info("Loading DIC files from disk")
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_pattern))

    all_data: dict[str, dict] = {}
    for f in files:
        try:
            slave_date, master_date = _parse_filename(
                f, filename_pattern=filename_pattern, date_format=date_format
            )
            date_str = slave_date.strftime(date_format)

            # Skip duplicates
            if date_str in all_data:
                continue

            # Compute time difference
            dt_hours = int(abs((slave_date - master_date).total_seconds()) / 3600)
            dt_days = int(dt_hours / 24)

            # If a preferred dt is set, filter by it (within tolerance). If None, keep all.
            if dt_days_preferred is not None and dt_days_preferred > 0:
                skip_file = abs(dt_days - int(dt_days_preferred)) > max(
                    0, int(dt_days_tolerance)
                )
                if skip_file:
                    continue

            # Read DIC data from file
            dic_data = _read_dic_file(f, invert_y=invert_y)
            dx = dic_data["dx"]
            dy = dic_data["dy"]
            disp_mag = dic_data["magnitude"]

            # Compute velocity (px/day) for dt > 24h, otherwise keep displacement as-is
            if dt_hours > 24:
                scale = 24.0 / dt_hours
                u_vel = dx * scale
                v_vel = dy * scale
                V_vel = disp_mag * scale
            else:
                u_vel = dx
                v_vel = dy
                V_vel = disp_mag

            all_data[date_str] = {
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
            }

        except Exception as e:
            logger.debug(f"Skipping {f.name}: {e}")
            continue

    logger.info(f"Loaded {len(all_data)} DIC files from disk")
    return all_data
