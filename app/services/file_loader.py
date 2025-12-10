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
        all_data = self.preload_all()
        return sorted(all_data.keys())

    def get_dic_data(self, date: str) -> dict | None:
        """Get data for a specific date."""
        all_data = self.preload_all()
        return all_data.get(date)

    def preload_all(self) -> dict[str, dict]:
        """Preload all DIC files from disk."""
        # Use cache if available
        if cache.all_data is not None:
            return cache.all_data

        if self._data_cache is not None:
            return self._data_cache

        logger.info("Preloading DIC files from disk")
        data_path = Path(self.settings.data_dir)
        files = sorted(data_path.glob(self.settings.file_pattern))

        all_data = {}

        for f in files:
            try:
                slave_date, master_date = self._parse_filename(f)
                date_str = slave_date.strftime(self.settings.date_format)

                dt_hours = int((slave_date - master_date).total_seconds() / 3600)
                if dt_hours <= 0:
                    dt_hours = abs(dt_hours)

                dt_days = int(dt_hours / 24)

                # Filter by preferred dt_days
                if abs(dt_days - self.settings.dt_days_preferred) > 1:
                    continue

                # Skip duplicates
                if date_str in all_data:
                    continue

                # Load file
                dic_data = self._read_file(f)

                # Compute velocity
                dx = dic_data["dx"]
                dy = dic_data["dy"]
                disp_mag = dic_data["magnitude"]

                if dt_hours > 24:
                    u_vel = dx / dt_hours * 24.0
                    v_vel = dy / dt_hours * 24.0
                    V_vel = disp_mag / dt_hours * 24.0
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
        cache.all_data = all_data
        self._data_cache = all_data
        return all_data

    def preload_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, dict]:
        """Preload only the dates in the requested range (files)."""
        # parse dates in YYYYMMDD or YYYY-MM-DD format

        def to_dateobj(s: str):
            if "-" in s:
                return datetime.strptime(s, "%Y-%m-%d").date()
            return datetime.strptime(s, "%Y%m%d").date()

        start_obj = to_dateobj(start_date)
        end_obj = to_dateobj(end_date)

        all_data = self.preload_all()
        filtered = {}
        dates = sorted(all_data.keys())
        # count selected
        selected_dates = [
            d
            for d in dates
            if start_obj <= datetime.strptime(d, "%Y%m%d").date() <= end_obj
        ]

        total = len(selected_dates)
        done = 0
        for d in selected_dates:
            filtered[d] = all_data[d]
            done += 1
            if progress_callback:
                progress_callback(done, total)

        # cache the filtered set
        cache.all_data = filtered
        self._data_cache = filtered
        return filtered

    def get_coordinates(self) -> np.ndarray:
        """Get coordinate array from first available date."""
        all_data = self.preload_all()
        if not all_data:
            raise ValueError("No data available")

        first_date = next(iter(all_data.keys()))
        first_data = all_data[first_date]
        return np.column_stack([first_data["x"], first_data["y"]])

    def _read_file(self, file_path: Path) -> dict[str, np.ndarray]:
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

        if self.settings.invert_y:
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

    def _parse_filename(self, file_path: Path) -> tuple[datetime, datetime]:
        """Parse dates from filename."""
        pattern = re.compile(self.settings.filename_pattern)
        match = pattern.search(file_path.name)

        if not match or len(match.groups()) < 2:
            raise ValueError(f"Invalid filename: {file_path.name}")

        slave_str = match.group(1)
        master_str = match.group(2)

        slave_date = datetime.strptime(slave_str, self.settings.date_format)
        master_date = datetime.strptime(master_str, self.settings.date_format)

        return slave_date, master_date
