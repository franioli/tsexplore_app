"""Database-backed DIC data provider with HTTP API fallback.

This provider tries to fetch DIC metadata and H5 file paths directly from the
Postgres database via psycopg. If the DB access fails (e.g., running outside
the compose network or DB unreachable), the provider falls back to fetch
DIC metadata and DIC arrays via the Django API using HTTP requests.

The returned shape is consistent with the FileDataProvider so the rest of the
app (plots, spatial indexing, timeseries) can remain unchanged.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psycopg
import requests
from requests.exceptions import RequestException

from ..cache import cache
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

USE_API = True  # Always allow API fallback


class DatabaseDataProvider:
    """Load DIC data from PostgreSQL database + H5 files, fallback to Django API."""

    def __init__(self):
        self.settings = get_settings()
        self._data_cache = None

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
    # Utility: HTTP API helpers
    # -------------------------
    def _api_base(self) -> str:
        """Return configured API base URL (host + port) or raise if not available."""
        # prefer explicit base url
        base = getattr(self.settings, "api_base_url", None)
        if base:
            return base.rstrip("/")
        host = getattr(self.settings, "api_host", None)
        port = getattr(self.settings, "api_port", None)
        if not host or not port:
            raise RuntimeError("API host/port not configured for fallback")
        return f"http://{host}:{port}"

    def _api_get_json(
        self, path: str, params: dict | None = None, timeout: int = 10
    ) -> Any:
        """Perform GET request to the Django API and return JSON (raises on HTTP error)."""
        base = self._api_base()
        url = (
            path
            if path.startswith("http")
            else f"{base.rstrip('/')}/{path.lstrip('/')}"
        )
        try:
            logger.debug(f"API request GET {url} params={params}")
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            logger.debug(f"API request failed GET {url}: {e}")
            raise

    # -------------------------
    # Conversion helpers
    # -------------------------
    def _parse_dic_json(self, data: dict) -> dict[str, np.ndarray | Any]:
        """Normalize JSON (from /API/dic/<id>/) to the same dict format as _read_h5() result."""
        # Expected keys: points Nx2, vectors Nx2 or None, magnitudes N
        points = np.array(data.get("points") or [], dtype=float)
        vectors = (
            np.array(data.get("vectors") or [], dtype=float)
            if data.get("vectors")
            else None
        )
        magnitudes = np.array(data.get("magnitudes") or [], dtype=float)

        if points.ndim == 1 and points.size == 2:
            points = points.reshape((1, 2))

        x = points[:, 0]
        y = points[:, 1]
        if vectors is None or vectors.size == 0:
            dx = np.zeros(len(points))
            dy = np.zeros(len(points))
        else:
            # vectors may be Nx2 arr
            dx = vectors[:, 0]
            dy = vectors[:, 1]

        # ensemble mad may be missing
        ensamble_mad = np.array(
            data.get("ensamble_mad") or np.zeros(len(points)), dtype=float
        )

        return {
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "magnitude": magnitudes,
            "ensamble_mad": ensamble_mad,
        }

    def _read_h5(self, h5_path: Path) -> dict[str, np.ndarray]:
        """Read H5 file content from disk (unchanged from previous behavior)."""
        with h5py.File(h5_path, "r") as f:
            points = np.array(f["points"])
            vectors = None
            if "vectors" in f:
                vectors = np.array(f["vectors"])
            magnitudes = (
                np.array(f["magnitudes"])
                if "magnitudes" in f
                else np.hypot(vectors[:, 0], vectors[:, 1])
            )
            ensamble_mad = np.array(f.get("ensamble_mad") or np.zeros(len(points)))

            return {
                "x": points[:, 0],
                "y": points[:, 1],
                "dx": vectors[:, 0] if vectors is not None else np.zeros(len(points)),
                "dy": vectors[:, 1] if vectors is not None else np.zeros(len(points)),
                "magnitude": magnitudes,
                "ensamble_mad": ensamble_mad,
            }

    # -------------------------
    # DB-backed methods
    # -------------------------
    def get_available_dates(self) -> list[str]:
        """Fetch available dates from database; fallback to API on failure."""
        try:
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
                    return [row[0].strftime("%Y%m%d") for row in rows]
        except Exception as db_exc:
            logger.warning(
                "DB access failed in get_available_dates(), falling back to API: %s",
                db_exc,
            )
            # try API fallback
            try:
                return self._get_available_dates_api()
            except Exception as api_exc:
                logger.error("API fallback failed for get_available_dates: %s", api_exc)
                # propagate the original DB error for visibility
                raise

    def get_dic_data(self, date: str) -> dict | None:
        """Fetch single DIC record for the given date (YYYYMMDD). DB first, API fallback."""
        # try DB path
        try:
            date_obj = datetime.strptime(date, "%Y%m%d").date()
        except ValueError:
            logger.error("Invalid date format: %s", date)
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, result_file_path, dt_hours
                        FROM ppcx_app_dic
                        WHERE reference_date = %s
                        AND ABS(dt_hours / 24.0 - %s) <= 1
                        ORDER BY ABS(dt_hours / 24.0 - %s)
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
                        logger.debug("No DIC row matched for date %s", date)
                        return None

                    dic_id, h5_path, dt_hours = row

                    if h5_path and Path(h5_path).exists():
                        dic_data = self._read_h5(Path(h5_path))
                    else:
                        # If DB has no h5_path or file missing, fallback to API to fetch content
                        dic_data = (
                            self._get_dic_h5_json_from_api(dic_id)
                            if (
                                hasattr(self.settings, "api_host")
                                or getattr(self.settings, "api_base_url", None)
                            )
                            else None
                        )
                        if not dic_data:
                            logger.warning(
                                "No H5 file found and API fallback unavailable for DIC id %s",
                                dic_id,
                            )
                            return None

                    # Compute velocity if needed
                    dx = dic_data["dx"]
                    dy = dic_data["dy"]
                    disp_mag = dic_data["magnitude"]

                    if dt_hours is None:
                        dt_hours = getattr(self.settings, "default_dt_hours", 24.0)

                    if dt_hours > 24:
                        u_vel = dx / dt_hours * 24.0
                        v_vel = dy / dt_hours * 24.0
                        V_vel = disp_mag / dt_hours * 24.0
                    else:
                        u_vel = dx
                        v_vel = dy
                        V_vel = disp_mag

                    return {
                        "x": dic_data["x"],
                        "y": dic_data["y"],
                        "dx": dx,
                        "dy": dy,
                        "disp_mag": disp_mag,
                        "u": u_vel,
                        "v": v_vel,
                        "V": V_vel,
                        "ensamble_mad": dic_data.get(
                            "ensamble_mad", np.zeros_like(disp_mag)
                        ),
                        "dt_hours": dt_hours,
                        "dt_days": int(dt_hours / 24),
                    }
        except Exception as db_exc:
            logger.warning(
                "DB access failed in get_dic_data(), trying API fallback: %s", db_exc
            )
            try:
                return self._get_dic_data_api(date)
            except Exception as api_exc:
                logger.error("API fallback failed for get_dic_data: %s", api_exc)
                raise

    def preload_all(self) -> dict[str, dict]:
        """Preload all DIC records; prefer DB, fallback to API if needed."""
        if cache.all_data is not None:
            return cache.all_data

        if self._data_cache is not None:
            return self._data_cache

        if not USE_API:
            logger.info("Preloading DIC from database")
            dates = self.get_available_dates()
            all_data = {}
            for date_str in dates:
                data = self.get_dic_data(date_str)
                if data is not None:
                    all_data[date_str] = data
            cache.all_data = all_data
            self._data_cache = all_data
            logger.info("Preloaded %d DIC records from DB", len(all_data))
            return all_data

        else:
            # fallback to API only
            try:
                dates = self._get_available_dates_api()
                all_data = {}
                for d in dates:
                    data = self._get_dic_data_api(d)
                    if data:
                        all_data[d] = data
                cache.all_data = all_data
                self._data_cache = all_data
                logger.info("Preloaded %d DIC records from API", len(all_data))
                return all_data
            except Exception as api_exc:
                logger.error("API preload failed: %s", api_exc)
                raise

    # -------------------------
    # API fallback implementations
    # -------------------------
    def _get_available_dates_api(self) -> list[str]:
        """Fetch list of available dates via the Django API (expects metadata or list)."""
        try:
            # Attempt to fetch DIC metadata; the endpoint typically returns a list of objects
            # including 'reference_date' field. Try a common path used by the Django API.
            data = self._api_get_json("/API/dic/")
        except Exception:
            # Try alternative path if the API is at /api/dic/
            try:
                data = self._api_get_json("/api/dic/")
            except Exception:
                raise

        dates = set()
        if isinstance(data, list):
            for item in data:
                rd = (
                    item.get("reference_date")
                    or item.get("referenceDate")
                    or item.get("reference")
                )
                if rd:
                    try:
                        dt = datetime.fromisoformat(rd)
                        dates.add(dt.strftime("%Y%m%d"))
                    except Exception:
                        # try YYYYMMDD directly
                        if isinstance(rd, str) and len(rd) == 8 and rd.isdigit():
                            dates.add(rd)
        else:
            # Data may be a dict with 'results' field (paginated API)
            results = data.get("results") if isinstance(data, dict) else None
            if isinstance(results, list):
                for item in results:
                    rd = item.get("reference_date")
                    if rd:
                        try:
                            dt = datetime.fromisoformat(rd)
                            dates.add(dt.strftime("%Y%m%d"))
                        except Exception:
                            if isinstance(rd, str) and len(rd) == 8 and rd.isdigit():
                                dates.add(rd)

        dates_list = sorted(dates)
        logger.info("API provided %d available dates", len(dates_list))
        return dates_list

    def _get_dic_h5_json_from_api(self, dic_id: int) -> dict | None:
        """Fetch /API/dic/<id>/ and return normalized dic content."""
        try:
            data = self._api_get_json(f"/API/dic/{dic_id}/")
        except Exception:
            try:
                data = self._api_get_json(f"/api/dic/{dic_id}/")
            except Exception:
                raise
        # Normalize into same dict format as _read_h5 returns
        return self._parse_dic_json(data)

    def _get_dic_data_api(self, date: str) -> dict | None:
        """Fetch dic data for the given date via API (search by reference_date then read DIC)."""
        # Try API filter parameter with iso date YYYY-MM-DD
        try:
            iso = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            iso = None

        # Primary attempt: use /API/dic/?reference_date=YYYY-MM-DD
        if iso:
            try:
                data = self._api_get_json("/API/dic/", params={"reference_date": iso})
            except Exception:
                try:
                    data = self._api_get_json(
                        "/api/dic/", params={"reference_date": iso}
                    )
                except Exception:
                    data = None
            if data:
                # if paginated, get results list
                if isinstance(data, dict) and "results" in data:
                    results = data["results"]
                elif isinstance(data, list):
                    results = data
                else:
                    results = []
                if results:
                    # pick best match (first)
                    dic_item = results[0]
                    dic_id = (
                        dic_item.get("dic_id")
                        or dic_item.get("id")
                        or dic_item.get("pk")
                    )
                    dt_hours = (
                        dic_item.get("dt_hours") or dic_item.get("dtHours") or None
                    )
                    if dic_id:
                        try:
                            dic_json = self._get_dic_h5_json_from_api(int(dic_id))
                            if dic_json:
                                dic_json["dt_hours"] = (
                                    dt_hours if dt_hours is not None else 24.0
                                )
                                return {
                                    "x": dic_json["x"],
                                    "y": dic_json["y"],
                                    "dx": dic_json["dx"],
                                    "dy": dic_json["dy"],
                                    "magnitude": dic_json["magnitude"],
                                    "ensamble_mad": dic_json.get(
                                        "ensamble_mad",
                                        np.zeros_like(dic_json["magnitude"]),
                                    ),
                                    "dt_hours": dic_json.get("dt_hours", 24.0),
                                }
                        except Exception:
                            logger.warning(
                                "API returned DIC id but retrieving content failed"
                            )
        # General fallback: GET /API/dic/ and look for matching reference_date field
        try:
            data = self._api_get_json("/API/dic/")
        except Exception:
            try:
                data = self._api_get_json("/api/dic/")
            except Exception:
                raise

        # Parse results similar to _get_available_dates_api
        candidates = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = data.get("results", [])

        for item in candidates:
            rd = item.get("reference_date")
            if not rd:
                continue
            try:
                rd_iso = datetime.fromisoformat(rd).strftime("%Y%m%d")
            except Exception:
                rd_iso = rd if isinstance(rd, str) else None
            if rd_iso == date:
                dic_id = item.get("dic_id") or item.get("id") or item.get("pk")
                if dic_id:
                    dic_json = self._get_dic_h5_json_from_api(int(dic_id))
                    if dic_json:
                        dt_hours = item.get("dt_hours") or 24.0
                        return {
                            "x": dic_json["x"],
                            "y": dic_json["y"],
                            "dx": dic_json["dx"],
                            "dy": dic_json["dy"],
                            "magnitude": dic_json["magnitude"],
                            "ensamble_mad": dic_json.get(
                                "ensamble_mad", np.zeros_like(dic_json["magnitude"])
                            ),
                            "dt_hours": dt_hours,
                        }
        # Nothing found via API
        return None
