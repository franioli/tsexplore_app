"""Database-backed DIC data provider.

This provider fetches the DIC id via a direct DB query using psycopg,
and then fetches the DIC displacements/points from the Django API.

Simple, fail-fast behavior: if DB or API calls fail, the exception will
propagate (no silent fallbacks). The API call is expected to return JSON
with at least 'points' and 'vectors'/'magnitudes'.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
import requests
from tqdm import tqdm

from ..cache import cache
from ..config import get_settings
from .data_provider import DataProvider

logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseDataProvider(DataProvider):
    """Load DIC data from PostgreSQL DB + Django API."""

    def __init__(self):
        self.settings = settings
        self._data_cache: dict[str, dict] | None = None

    # -------------------------
    # Utility: DB connection
    # -------------------------
    def _get_connection(self):
        """Create a DB connection (read-only)."""
        return psycopg.connect(
            host=self.settings.db_host,
            port=self.settings.db_port,
            dbname=self.settings.db_name,
            user=self.settings.db_user,
            password=self.settings.db_password,
            options="-c default_transaction_read_only=on",
        )

    # -------------------------
    # Helper: API host/port
    # -------------------------
    def _api_host_port(self) -> tuple[str, int]:
        """
        Resolve the Django API host and port from settings.
        Falls back to sensible defaults for the compose environment.
        """
        host = getattr(self.settings, "api_host", None) or "localhost"
        port = int(getattr(self.settings, "api_port", None) or 8000)
        return host, port

    # -------------------------
    # Main Provider methods
    # -------------------------
    def get_available_dates(self) -> list[str]:
        """Fetch distinct reference_date values (YYYYMMDD) from DB."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT DISTINCT reference_date
                    FROM ppcx_app_dic
                    WHERE ABS(dt_hours / 24.0 - %s) <= 1
                    ORDER BY reference_date
                """
                cur.execute(query, (self.settings.dt_days_preferred,))
                rows = cur.fetchall()
                return [r[0].strftime("%Y%m%d") for r in rows]

    def load_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, dict]:
        """
        Load DIC data for a given date range. Calls progress_callback(done,total).
        Fail fast (exceptions bubble).
        """

        def to_dateobj(s: str):
            if "-" in s:
                return datetime.strptime(s, "%Y-%m-%d").date()
            return datetime.strptime(s, "%Y%m%d").date()

        start_obj = to_dateobj(start_date)
        end_obj = to_dateobj(end_date)
        if start_obj > end_obj:
            raise ValueError("start_date must be <= end_date")

        # fetch dates within range
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT DISTINCT DATE(reference_date)
                    FROM ppcx_app_dic
                    WHERE DATE(reference_date) BETWEEN %s AND %s
                    AND ABS(dt_hours / 24.0 - %s) <= 1
                    ORDER BY reference_date
                """
                cur.execute(
                    query, (start_obj, end_obj, self.settings.dt_days_preferred)
                )
                rows = cur.fetchall()
                dates = [r[0].strftime("%Y%m%d") for r in rows]

        total = len(dates)
        done = 0
        all_data: dict[str, dict] = {}

        # update cache load metadata
        cache.load_in_progress = True
        cache.load_total = total
        cache.load_done = done
        cache.load_start_date = start_date
        cache.load_end_date = end_date
        cache.load_error = None

        for idx, d in enumerate(dates):
            # for each date fetch its dic
            ddata = self.get_dic_data(d)
            if ddata is not None:
                all_data[d] = ddata

            done += 1
            cache.load_done = done
            if progress_callback:
                progress_callback(done, total)

        # finalize
        cache.all_data = all_data
        cache.load_in_progress = False
        cache.load_total = total
        cache.load_done = done

        # build kdtree in-memory (allow spatial queries)
        from .spatial import build_kdtree

        build_kdtree(self)

        return all_data

    def load_all(self) -> dict[str, dict]:
        """Load all available DIC data into memory and cache it."""
        if cache.all_data is not None:
            return cache.all_data
        if self._data_cache is not None:
            return self._data_cache

        dates = self.get_available_dates()
        all_data: dict[str, dict] = {}
        for d in tqdm(dates, desc="Loading DIC data from DB"):
            data = self.get_dic_data(d)
            if data is not None:
                all_data[d] = data

        cache.all_data = all_data
        self._data_cache = all_data
        return all_data

    def get_coordinates(self) -> np.ndarray:
        """Return Nx2 array of coordinates used to build KDTree."""
        # rely on load_all to fill cache; this will raise if there's no data
        all_data = self.load_all()
        if not all_data:
            raise RuntimeError("No DIC data available")
        first = next(iter(all_data.values()))
        return np.column_stack([first["x"], first["y"]])

    def get_dic_data(self, date: str) -> dict[str, Any] | None:
        """
        Query DB for the best DIC for the requested date, then fetch the points/vectors
        via the Django API for that dic_id.

        Returns the same dict shape as FileDataProvider for compatibility:
        {
          "x": np.ndarray,
          "y": np.ndarray,
          "dx": np.ndarray,  # displacement EW
          "dy": np.ndarray,  # displacement NS
          "disp_mag": np.ndarray,
          "u": np.ndarray,   # velocity EW (px/day)
          "v": np.ndarray,   # velocity NS (px/day)
          "V": np.ndarray,   # velocity magnitude (px/day)
          "ensemble_mad": np.ndarray,
          "dt_hours": int,
          "dt_days": int,
        }
        """
        # normalize date parameter
        try:
            date_obj = datetime.strptime(date, "%Y%m%d").date()
        except ValueError:
            raise ValueError(f"Invalid date format: {date}, expected YYYYMMDD")

        # Query DB for dic id and dt_hours
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT id, dt_hours
                    FROM ppcx_app_dic
                    WHERE DATE(reference_date) = %s
                    AND ABS(dt_hours / 24.0 - %s) <= 1
                    ORDER BY ABS(dt_hours / 24.0 - %s) ASC
                    LIMIT 1
                """
                cur.execute(
                    query,
                    (
                        date_obj,
                        self.settings.dt_days_preferred,
                        self.settings.dt_days_preferred,
                    ),
                )
                row = cur.fetchone()
                if not row:
                    return None
                dic_id, dt_hours = row
                dt_hours = int(dt_hours)
                dt_days = int(dt_hours / 24)

        # Fetch DIC points/vectors from Django API
        host, port = self._api_host_port()
        url = (
            f"http://{host}:{port}/API/dic/{dic_id}/"  # matches the project's API route
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()  # fail fast on non-200

        payload = resp.json()

        # Validate structure
        if "points" not in payload:
            raise ValueError(f"Invalid API response for dic {dic_id}: missing 'points'")

        points = payload["points"]
        if not points:
            return None

        # Vectors may be under keys 'vectors' or 'displacements' or 'uv'
        vectors = (
            payload.get("vectors") or payload.get("displacements") or payload.get("uv")
        )
        magnitudes = (
            payload.get("magnitudes") or payload.get("V") or payload.get("magnitude")
        )

        # Build numpy arrays
        pts = np.array(points, dtype=float)
        x = pts[:, 0]
        y = (
            -pts[:, 1] if self.settings.invert_y else pts[:, 1]
        )  # apply invert_y like file loader

        if vectors is None:
            # if vectors missing but magnitudes present, set zero vectors
            dx = np.zeros_like(x)
            dy = np.zeros_like(y)
        else:
            vecs = np.array(vectors, dtype=float)
            dx = vecs[:, 0]
            dy = vecs[:, 1]
            if self.settings.invert_y:
                dy = -dy

        # displacement magnitude (ensure available)
        if magnitudes is None:
            disp_mag = np.hypot(dx, dy)
        else:
            disp_mag = np.array(magnitudes, dtype=float)

        # compute velocities (px/day) using dt_hours
        if dt_hours and dt_hours > 0:
            u_vel = dx / dt_hours * 24.0
            v_vel = dy / dt_hours * 24.0
            V_vel = disp_mag / dt_hours * 24.0
        else:
            u_vel = dx
            v_vel = dy
            V_vel = disp_mag

        ensemble_mad = np.array(
            payload.get("ensemble_mad")
            or payload.get("ensemble_mad")
            or payload.get("mad")
            or np.zeros_like(disp_mag),
            dtype=float,
        )

        return {
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "disp_mag": disp_mag,
            "u": u_vel,
            "v": v_vel,
            "V": V_vel,
            "ensemble_mad": ensemble_mad,
            "dt_hours": dt_hours,
            "dt_days": dt_days,
        }
