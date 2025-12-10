"""Database access for DIC results (read-only)."""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import psycopg  # or use Django ORM if you add Django as dependency

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_db_connection():
    """Create a database connection using settings."""
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        # Read-only connection
        options="-c default_transaction_read_only=on",
    )


def fetch_dic_by_date(
    date_str: str, dt_days_preferred: int = 1
) -> dict[str, Any] | None:
    """
    Fetch a single DIC record for a given date from the database.

    Args:
        date_str: Date in YYYYMMDD format
        dt_days_preferred: Preferred time difference in days

    Returns:
        Dictionary with DIC data or None if not found
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        logger.error(f"Invalid date format: {date_str}")
        return None

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Query for DIC with matching reference_date and dt_hours close to preferred
            query = """
                SELECT 
                    id, 
                    result_file_path,
                    dt_hours,
                    master_timestamp,
                    slave_timestamp
                FROM ppcx_app_dic
                WHERE reference_date = %s
                AND ABS(dt_hours / 24.0 - %s) <= 1
                ORDER BY ABS(dt_hours / 24.0 - %s)
                LIMIT 1
            """
            cur.execute(query, (date_obj, dt_days_preferred, dt_days_preferred))
            row = cur.fetchone()

            if not row:
                return None

            dic_id, h5_path, dt_hours, master_ts, slave_ts = row

            # Read H5 file
            from pathlib import Path

            try:
                # Try to read from result_file_path (H5 format)
                if h5_path and Path(h5_path).exists():
                    dic_data = read_h5_dic(Path(h5_path))
                else:
                    logger.warning(f"H5 file not found for DIC {dic_id}, skipping")
                    return None

            except Exception as e:
                logger.error(f"Failed to read DIC data for {date_str}: {e}")
                return None

            # Compute velocity from displacement
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

            return {
                "dic_id": dic_id,
                "x": dic_data["x"],
                "y": dic_data["y"],
                "dx": dx,
                "dy": dy,
                "disp_mag": disp_mag,
                "u": u_vel,
                "v": v_vel,
                "V": V_vel,
                "ensamble_mad": dic_data.get("ensamble_mad", np.zeros_like(disp_mag)),
                "dt_hours": dt_hours,
                "dt_days": int(dt_hours / 24),
            }


def fetch_available_dates(dt_days_preferred: int = 1) -> list[str]:
    """
    Fetch all available dates from the database.

    Args:
        dt_days_preferred: Preferred time difference in days

    Returns:
        List of date strings in YYYYMMDD format
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT DISTINCT reference_date
                FROM ppcx_app_dic
                WHERE ABS(dt_hours / 24.0 - %s) <= 1
                ORDER BY reference_date
            """
            cur.execute(query, (dt_days_preferred,))
            rows = cur.fetchall()

            return [row[0].strftime("%Y%m%d") for row in rows]


def read_h5_dic(h5_path) -> dict[str, np.ndarray]:
    """Read DIC H5 file (same format as Django app uses)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return {
            "x": f["points"][:, 0],
            "y": f["points"][:, 1],
            "dx": f["vectors"][:, 0] if "vectors" in f else np.zeros(len(f["points"])),
            "dy": f["vectors"][:, 1] if "vectors" in f else np.zeros(len(f["points"])),
            "magnitude": f["magnitudes"][:],
            "ensamble_mad": f.get("ensamble_mad", np.zeros(len(f["points"]))),
        }
