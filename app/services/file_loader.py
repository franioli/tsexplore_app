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
from tqdm import tqdm

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
            file_search_pattern=s.file_search_pattern,
            filename_date_template=getattr(s, "filename_date_template", None),
            # filename_pattern=s.filename_pattern,
            # date_format=s.date_format,
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
    file_path: Path,
    *,
    filename_date_template: str | None,
    filename_pattern: str | None = None,
    date_format: str | None = None,
) -> tuple[datetime, datetime]:
    """Parse initial/final dates from a DIC filename.

    Supports two modes:
    - filename_date_template: a template matching the file stem with placeholders
      {final:<strftime>} and {initial:<strftime>} (preferred).
    - filename_pattern + date_format: legacy regex (two capture groups) + single strptime format.

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
    stem = file_path.stem

    # find placeholders in order
    if filename_date_template:
        ph_re = re.compile(r"\{(final|initial):([^}]+)\}")
        placeholders = []
        last = 0
        pattern_parts = []

        def _escape_literal_with_glob(lit: str) -> str:
            # escape all regex metachars then replace escaped glob tokens back to regex
            esc = re.escape(lit)
            # allow '*' and '?' in template to act as glob wildcards
            esc = esc.replace(r"\*", ".*").replace(r"\?", ".")
            return esc

        for m in ph_re.finditer(filename_date_template):
            placeholders.append((m.group(1), m.group(2)))
            # escape literal between placeholders but keep '*' and '?' as wildcards
            literal = filename_date_template[last : m.start()]
            pattern_parts.append(_escape_literal_with_glob(literal))

            # convert strftime -> simple regex
            fmt = m.group(2)

            def _strftime_to_regex(s: str) -> str:
                mapping = {
                    "%Y": r"\d{4}",
                    "%y": r"\d{2}",
                    "%m": r"\d{2}",
                    "%d": r"\d{2}",
                    "%H": r"\d{2}",
                    "%M": r"\d{2}",
                    "%S": r"\d{2}",
                    "%f": r"\d+",
                    "%j": r"\d{3}",
                }
                return re.sub(
                    r"%[A-Za-z]", lambda mm: mapping.get(mm.group(0), r".+?"), s
                )

            pattern_parts.append(f"({_strftime_to_regex(fmt)})")
            last = m.end()

        # tail literal (may contain '*'/'?')
        pattern_parts.append(_escape_literal_with_glob(filename_date_template[last:]))
        full_re = "^" + "".join(pattern_parts) + "$"
        m = re.match(full_re, stem)
        if not m:
            raise ValueError(f"Filename does not match template: {file_path.name}")

        groups = list(m.groups())
        vals = {}
        for (ph, fmt), val in zip(placeholders, groups):
            try:
                vals[ph] = datetime.strptime(val, fmt)
            except Exception as e:
                raise ValueError(f"Failed parsing '{val}' with format '{fmt}': {e}")

        if "final" not in vals or "initial" not in vals:
            raise ValueError("Template must contain both {final:...} and {initial:...}")

        return vals["final"], vals["initial"]

    # Fallback to legacy regex + single date_format
    if not filename_pattern or not date_format:
        raise ValueError("No valid filename parsing configuration provided")

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
    """Faster HDF5 reader: read only needed datasets and use read_direct to avoid extra copies."""
    import h5py

    with h5py.File(file_path, "r") as f:
        # pick ensemble group or first pair group
        if "ensemble" in f:
            g = f["ensemble"]
        else:
            groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
            if not groups:
                raise ValueError("HDF5 file contains no data groups")
            g = f[sorted(groups)[0]]

        # determine number of nodes
        if "nodes" in g:
            n_nodes = g["nodes"].shape[0]
        else:
            # fallback: try dx length or zero
            n_nodes = int(g.get("dx", np.empty(0)).shape[0])

        # helper to read a dataset into preallocated array (if present)
        def _read_ds(name, dtype=np.float32, shape=None):
            if name not in g:
                return np.zeros(shape or (n_nodes,), dtype=dtype)
            ds = g[name]
            out = np.empty(ds.shape, dtype=ds.dtype)
            ds.read_direct(out)
            return out

        nodes = _read_ds("nodes", dtype=np.float32)
        if nodes.ndim == 2 and nodes.shape[1] >= 2:
            x = nodes[:, 0].astype(np.float32)
            y = nodes[:, 1].astype(np.float32)
        else:
            x = _read_ds("x", dtype=np.float32, shape=(n_nodes,))
            y = _read_ds("y", dtype=np.float32, shape=(n_nodes,))

        dx = _read_ds("dx", dtype=np.float32, shape=(n_nodes,))
        dy = _read_ds("dy", dtype=np.float32, shape=(n_nodes,))

        # prefer explicit 'mad' or fall back to 'corr' or zeros
        if "mad" in g:
            ensamble_mad = _read_ds("mad", dtype=np.float32, shape=(n_nodes,))
        elif "corr" in g:
            ensamble_mad = _read_ds("corr", dtype=np.float32, shape=(n_nodes,))
        else:
            ensamble_mad = np.zeros_like(dx)

        # ensure 1-D and length-consistent
        x = np.asarray(x, dtype=np.float32).ravel()[:n_nodes]
        y = np.asarray(y, dtype=np.float32).ravel()[:n_nodes]
        dx = np.asarray(dx, dtype=np.float32).ravel()[:n_nodes]
        dy = np.asarray(dy, dtype=np.float32).ravel()[:n_nodes]
        ensamble_mad = np.asarray(ensamble_mad, dtype=np.float32).ravel()[:n_nodes]

        if invert_y:
            y = -y
            dy = -dy

        magnitude = np.hypot(dx, dy)

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
        data = _read_h5(file_path, invert_y)
    elif suffix == ".txt":
        data = _read_txt(file_path, invert_y)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    return data


def _load_all_from_disk_into_cache(
    *,
    data_dir: str | Path,
    file_search_pattern: str = "*.txt",
    filename_date_template: str | None,
    filename_pattern: str | None = None,
    date_format: str | None = None,
    invert_y: bool = False,
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
    files = sorted(data_path.glob(file_search_pattern))

    loaded = 0
    for f in tqdm(files, desc="Loading DIC files", unit="file"):
        try:
            final_date, initial_date = _parse_filename(
                f,
                filename_date_template=filename_date_template,
                filename_pattern=filename_pattern,
                date_format=date_format,
            )

            # Compute time difference
            dt_hours = int(abs((final_date - initial_date).total_seconds()) / 3600)
            dt_days = int(dt_hours / 24)

            # Optional filtering by preferred dt
            if dt_days_preferred is not None and int(dt_days_preferred) > 0:
                skip_file = abs(dt_days - int(dt_days_preferred)) > max(
                    0, int(dt_days_tolerance)
                )
                if skip_file:
                    continue

            # Read DIC data
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
            logger.warning(f"Skipping {f.name}: {e}")
            continue

    logger.info(f"Loaded {loaded} DIC files from disk (records={cache.num_records})")
    return loaded
