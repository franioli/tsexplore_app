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

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..cache import DataCache
from ..cache import cache as global_cache
from ..config import get_settings
from .data_provider import DataProvider

logger = logging.getLogger(__name__)


class FileDataProvider(DataProvider):
    """Load DIC data from text files on disk."""

    def __init__(self, cache: DataCache | None = None):
        """Initialize the file-backed provider."""

        self.settings = get_settings()

        # allow cache injection; default to module-level global cache for backwards compat
        self._cache = cache if cache is not None else global_cache

    def bind_cache(self, cache: DataCache):
        """Optional helper to set/replace the cache after construction."""
        self._cache = cache

    def get_available_dates(self) -> list[str]:
        """Return all available reference dates (YYYY-MM-DD) without loading DIC arrays.

        For file backend this means scanning filenames and parsing {final}/{initial}
        from the stem using filename_date_template.
        """
        s = self.settings
        data_path = Path(s.data_dir)
        files = data_path.glob(s.file_search_pattern)

        # date patten in filenames
        filename_date_template = getattr(s, "filename_date_template", None)

        # legacy args kept for compatibility
        filename_pattern = getattr(s, "filename_pattern", None)
        date_format = getattr(s, "date_format", None)

        dates: set[str] = set()
        for f in files:
            try:
                final_date, _initial_date = _parse_filename(
                    f,
                    filename_date_template=filename_date_template,
                    filename_pattern=filename_pattern,
                    date_format=date_format,
                )
                dates.add(final_date.strftime("%Y-%m-%d"))
            except Exception as e:
                logger.warning(f"Skipping file {f.name}: {e}")
                continue

        out = sorted(dates)
        return out

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
            reference_date: Reference date in YYYY-MM-DD (final image date).
            dt_days: If provided, select the record with this time interval (days).
            initial_date: If provided, select by initial date (YYYY-MM-DD). Can be combined with dt_days for an exact match.
            prefer_dt_days: If provided, pick the record whose dt_days is closest.
            prefer_dt_tolerance: Optional tolerance (days) for closest match.

        Returns:
            The payload dict for the selected record, or None if no matching record exists.
        """
        if not self._cache.is_loaded():
            raise ValueError(
                "No data loaded yet. Press 'Load' before requesting DIC data."
            )

        # Convert reference_date to YYYYMMDD for cache lookup
        if "-" in reference_date:
            reference_date = datetime.strptime(reference_date, "%Y-%m-%d").strftime(
                "%Y%m%d"
            )
        if initial_date and "-" in initial_date:
            initial_date = datetime.strptime(initial_date, "%Y-%m-%d").strftime(
                "%Y%m%d"
            )

        rid = self._cache.find_record_id(
            reference_yyyymmdd=reference_date,
            initial_yyyymmdd=initial_date,
            dt_days=dt_days,
            prefer_dt_days=prefer_dt_days,
            prefer_dt_tolerance=prefer_dt_tolerance,
        )
        if rid is None:
            return None

        rec = self._cache.get_dic_record(rid)
        if rec is None:
            return None
        _meta, payload = rec
        return payload

    def load_all(self) -> int:
        """Load all DIC files from disk into the global cache.

        Returns:
            The number of loaded records (one record per file).
        """
        if self._cache.is_loaded():
            return self._cache.num_records

        data_dir = self.settings.data_dir

        search_pattern = self.settings.file_search_pattern
        filename_date_template = getattr(self.settings, "filename_date_template", None)
        filename_pattern = getattr(self.settings, "filename_pattern", None)
        date_format = getattr(self.settings, "date_format", None)

        # dt days preferred setting
        dt_days = getattr(self.settings, "dt_days", None)
        dt_hours_tolerance = getattr(self.settings, "dt_hours_tolerance", 0)

        invert_y = self.settings.invert_y

        n = _load_data_to_cache(
            data_dir=data_dir,
            file_search_pattern=search_pattern,
            filename_date_template=filename_date_template,
            filename_pattern=filename_pattern,
            date_format=date_format,
            invert_y=invert_y,
            dt_days=dt_days,
            dt_hours_tolerance=dt_hours_tolerance,
            cache=self._cache,
        )
        self._loaded = True
        return n

    def load_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Load only files whose FINAL date is in [start_date, end_date] (inclusive).

        Dates are expected as 'YYYY-MM-DD' (from UI).
        """

        data_dir = self.settings.data_dir

        search_pattern = self.settings.file_search_pattern
        filename_date_template = getattr(self.settings, "filename_date_template", None)
        filename_pattern = getattr(self.settings, "filename_pattern", None)
        date_format = getattr(self.settings, "date_format", None)

        # dt days preferred setting
        dt_days = getattr(self.settings, "dt_days", None)
        dt_hours_tolerance = getattr(self.settings, "dt_hours_tolerance", 0)

        invert_y = self.settings.invert_y

        dt_start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        dt_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        n = _load_data_to_cache(
            data_dir=data_dir,
            file_search_pattern=search_pattern,
            filename_date_template=filename_date_template,
            filename_pattern=filename_pattern,
            date_format=date_format,
            reference_date_start=dt_start,
            reference_date_end=dt_end,
            invert_y=invert_y,
            dt_days=dt_days,
            dt_hours_tolerance=dt_hours_tolerance,
            progress_callback=progress_callback,
            cache=self._cache,
        )
        return n

    def get_coordinates(self) -> np.ndarray:
        """Get coordinate array from the first loaded record.

        Returns:
            (N, 2) array with columns [x, y].

        Raises:
            ValueError: If no data is available.
        """
        if not self._cache.is_loaded():
            self.load_all()

        if self._cache.num_records == 0:
            raise ValueError("No data available")

        # Pick the first record_id deterministically
        first_id = sorted(self._cache.dic_records.keys())[0]
        rec = self._cache.get_dic_record(first_id)
        if rec is None:
            raise ValueError("No data available")

        _meta, payload = rec
        return np.column_stack([payload["x"], payload["y"]])

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
        if not self._cache.is_loaded():
            raise ValueError(
                "No data loaded yet. Press 'Load' before requesting timeseries."
            )
        if self._cache.kdtree is None:
            raise ValueError(
                "No kdtree is available in data cache. Try reloading the data."
            )

        # Find nearest node to the selected coordinates
        tree, _ = self._cache.kdtree
        dist, idx = tree.query([node_x, node_y], k=1)
        if not np.isfinite(dist):
            logger.warning(f"Node ({node_x}, {node_y}) not found in KDTree")
            return {}

        # Accumulate lists for a single timeseries
        ref_dates: list[Any] = []
        init_dates: list[Any] = []
        final_dates: list[Any] = []
        dt_days_list: list[int] = []
        dx_list: list[Any] = []
        dy_list: list[Any] = []
        disp_mag_list: list[Any] = []
        u_list: list[Any] = []
        v_list: list[Any] = []
        V_list: list[Any] = []
        mad_list: list[Any] = []

        def get_value(
            array: np.ndarray, idx: int, record_name: str | None = None
        ) -> Any:
            try:
                array_value = array[idx]
            except IndexError:
                logger.warning(
                    f"Index {idx} is out of bounds for record field '{record_name}' with length {len(array)}"
                )
                array_value = np.nan
            return array_value

        # Iterate over all records in the cache (sorted by record_id) and filter by dt_days+tolerance
        for meta, record in self._cache:
            rec_dt_hours = (
                meta.final_date - meta.initial_date
            ).total_seconds() / 3600.0
            if dt_days is not None:
                target_hours = float(dt_days) * 24.0
                if abs(rec_dt_hours - target_hours) > dt_hours_tolerance:
                    # skip if outside tolerance
                    continue

            # keep this record
            ref_dates.append(meta.final_date)
            init_dates.append(meta.initial_date)
            final_dates.append(meta.final_date)
            dt_days_list.append(int(round(rec_dt_hours / 24.0)))
            dx_list.append(get_value(record["dx"], idx, "dx"))
            dy_list.append(get_value(record["dy"], idx, "dy"))
            disp_mag_list.append(get_value(record["disp_mag"], idx, "disp_mag"))
            u_list.append(get_value(record["u"], idx, "u"))
            v_list.append(get_value(record["v"], idx, "v"))
            V_list.append(get_value(record["V"], idx, "V"))
            if "ensemble_mad" in record and record["ensemble_mad"] is not None:
                mad_list.append(get_value(record["ensemble_mad"], idx, "ensemble_mad"))
            else:
                mad_list.append(None)

        # Convert to numpy arrays, sort by reference date
        if len(ref_dates) == 0:
            return {}

        ref_arr = np.asarray(ref_dates, dtype="datetime64[D]")
        order = np.argsort(ref_arr)

        return {
            "reference_dates": ref_arr[order],
            "initial_dates": np.asarray(init_dates, dtype="datetime64[D]")[order],
            "final_dates": np.asarray(final_dates, dtype="datetime64[D]")[order],
            "dt_days": np.asarray(dt_days_list, dtype=np.int32)[order],
            "dx": np.asarray(dx_list, dtype=np.float32)[order],
            "dy": np.asarray(dy_list, dtype=np.float32)[order],
            "disp_mag": np.asarray(disp_mag_list, dtype=np.float32)[order],
            "u": np.asarray(u_list, dtype=np.float32)[order],
            "v": np.asarray(v_list, dtype=np.float32)[order],
            "V": np.asarray(V_list, dtype=np.float32)[order],
            "ensemble_mad": np.asarray(
                [v if v is not None else np.nan for v in mad_list], dtype=np.float32
            )[order],
        }


@lru_cache(maxsize=64)
def _compile_filename_date_template(
    filename_date_template: str,
) -> tuple[re.Pattern[str], list[tuple[str, str]]]:
    """Compile filename_date_template into a regex and a placeholder list.

    Supports glob wildcards in literals:
    - '*' => '.*'
    - '?' => '.'
    """
    ph_re = re.compile(r"\{(final|initial):([^}]+)\}")
    placeholders: list[tuple[str, str]] = []
    last = 0
    pattern_parts: list[str] = []

    def _escape_literal_with_glob(lit: str) -> str:
        esc = re.escape(lit)
        return esc.replace(r"\*", ".*").replace(r"\?", ".")

    def _strftime_to_regex(fmt: str) -> str:
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
        return re.sub(r"%[A-Za-z]", lambda mm: mapping.get(mm.group(0), r".+?"), fmt)

    for m in ph_re.finditer(filename_date_template):
        placeholders.append((m.group(1), m.group(2)))

        literal = filename_date_template[last : m.start()]
        pattern_parts.append(_escape_literal_with_glob(literal))

        fmt = m.group(2)
        pattern_parts.append(f"({_strftime_to_regex(fmt)})")

        last = m.end()

    pattern_parts.append(_escape_literal_with_glob(filename_date_template[last:]))

    full_re = "^" + "".join(pattern_parts) + "$"
    return re.compile(full_re), placeholders


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
        rx, placeholders = _compile_filename_date_template(filename_date_template)
        m = rx.match(stem)
        if not m:
            raise ValueError(
                f"Filename {file_path.name} does not match template: {filename_date_template}"
            )

        groups = list(m.groups())
        vals: dict[str, datetime] = {}
        for (ph, fmt), val in zip(placeholders, groups):
            vals[ph] = datetime.strptime(val, fmt)

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
        ``ensemble_mad``.

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
    ensemble_mad = data[:, 4] if data.shape[1] > 4 else np.zeros_like(magnitude)

    return {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
        "ensemble_mad": ensemble_mad,
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
            ensemble_mad = _read_ds("mad", dtype=np.float32, shape=(n_nodes,))
        elif "corr" in g:
            ensemble_mad = _read_ds("corr", dtype=np.float32, shape=(n_nodes,))
        else:
            ensemble_mad = np.zeros_like(dx)

        # ensure 1-D and length-consistent
        x = np.asarray(x, dtype=np.float32).ravel()[:n_nodes]
        y = np.asarray(y, dtype=np.float32).ravel()[:n_nodes]
        dx = np.asarray(dx, dtype=np.float32).ravel()[:n_nodes]
        dy = np.asarray(dy, dtype=np.float32).ravel()[:n_nodes]
        ensemble_mad = np.asarray(ensemble_mad, dtype=np.float32).ravel()[:n_nodes]

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
        "ensemble_mad": ensemble_mad,
    }


def _read_single_file(
    path: str,
    final_date: str,
    initial_date: str,
    dt_hours: int,
    dt_days: int,
    invert_y: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a single DIC file and prepare the metadata and payload for caching.

    This is a top-level, picklable worker intended for use with
    concurrent.futures.ProcessPoolExecutor. It reads the given DIC file
    (either HDF5 or text), computes displacement magnitude and velocities
    (scaled when dt_hours > 24), and constructs the ``meta`` and ``payload``
    dictionaries expected by the cache.

    Args:
        f_str: Path to the DIC file (string). Must exist.
        final_iso: Final (reference) date as ISO-formatted string (YYYY-MM-DD or ISO).
        initial_iso: Initial date as ISO-formatted string.
        dt_hours: Time difference between images in hours.
        dt_days: Time difference between images in days.
        invert_y: If True, invert the y coordinate and dy sign.

    Returns:
        A tuple (meta, payload):
            meta (dict): contains datetime objects and dt metadata:
                {
                  "initial_date": datetime,
                  "final_date": datetime,
                  "reference_date": datetime,  # same as final_date
                  "dt_hours": int,
                  "dt_days": int
                }
            payload (dict): contains numpy arrays and auxiliary metadata:
                {
                  "x", "y", "dx", "dy", "disp_mag", "u", "v", "V",
                  "ensemble_mad", "dt_hours", "dt_days",
                  "initial_date", "final_date", "reference_date"
                }

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has an unsupported extension or is invalid.
        Any IO / parsing errors raised while reading the file are propagated.
    """
    f = Path(path)
    if not f.exists():
        raise FileNotFoundError(f"File not found: {f}")

    final_date_ = datetime.fromisoformat(final_date)
    initial_date_ = datetime.fromisoformat(initial_date)

    # Read the file according to its extension
    suffix = f.suffix.lower()
    if suffix == ".h5":
        dic_data = _read_h5(f, invert_y)
    elif suffix == ".txt":
        dic_data = _read_txt(f, invert_y)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    dx = dic_data["dx"]
    dy = dic_data["dy"]
    disp_mag = dic_data["magnitude"]

    # If dt > 24 hours, convert displacements to velocity (px/day)
    if dt_hours > 24:
        scale = 24.0 / float(dt_hours)
        u_vel = dx * scale
        v_vel = dy * scale
        V_vel = disp_mag * scale
    else:
        u_vel = dx
        v_vel = dy
        V_vel = disp_mag

    meta: dict[str, Any] = {
        "initial_date": initial_date_,
        "final_date": final_date_,
        "reference_date": final_date_,
        "dt_hours": int(dt_hours),
        "dt_days": int(dt_days),
    }

    payload: dict[str, Any] = {
        "x": dic_data["x"],
        "y": dic_data["y"],
        "dx": dx,
        "dy": dy,
        "disp_mag": disp_mag,
        "u": u_vel,
        "v": v_vel,
        "V": V_vel,
        "ensemble_mad": dic_data.get("ensemble_mad"),
        "dt_hours": int(dt_hours),
        "dt_days": int(dt_days),
        "initial_date": initial_date_.strftime("%Y-%m-%d"),
        "final_date": final_date_.strftime("%Y-%m-%d"),
        "reference_date": final_date_.strftime("%Y-%m-%d"),
    }
    return meta, payload


def _load_data_to_cache(
    *,
    data_dir: str | Path,
    file_search_pattern: str = "*.txt",
    filename_date_template: str | None,
    filename_pattern: str | None = None,
    date_format: str | None = None,
    invert_y: bool = False,
    dt_days: int | list[int] | None = None,
    dt_hours_tolerance: int = 0,
    reference_date_start: datetime | None = None,
    reference_date_end: datetime | None = None,
    cache: DataCache = global_cache,
    progress_callback: Callable[[int, int], None] | None = None,
    max_workers: int | None = 1,
) -> int:
    """Load all DIC files from disk into the global cache (one record per file).

    If final_date_start/end are provided, only files whose FINAL date is within
    [final_date_start, final_date_end] (inclusive, date-based) are loaded.

    Args:
        data_dir: Root directory containing DIC files.
        file_pattern: Glob pattern used to list files (e.g. ``"*.txt"``).
        filename_pattern: Regex pattern extracting slave/master date strings.
        date_format: Format string for parsing the extracted date strings.
        invert_y: Whether to invert the y-axis (and dy) sign.
        dt_days: If set (> 0), keep only records within ``dt_days_tolerance``
            from the preferred dt. If ``None``, all records are kept.
        dt_days_tolerance: Allowed absolute deviation from ``dt_days`` in days.

    Returns:
        Number of loaded records stored into the global cache.
    """
    logger.info("Loading DIC files from disk..")
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_search_pattern))

    # Pre-filter files by parsing filenames and applying date / dt filters
    candidates: list[tuple[Path, datetime, datetime, int, int]] = []
    for f in files:
        try:
            final_date, initial_date = _parse_filename(
                f,
                filename_date_template=filename_date_template,
                filename_pattern=filename_pattern,
                date_format=date_format,
            )
        except Exception:
            logger.error(f"Skipping {f.name}: filename parsing failed")
            continue

        # Filter by Reference date
        if reference_date_start and final_date.date() < reference_date_start.date():
            continue
        if reference_date_end and final_date.date() > reference_date_end.date():
            continue

        # Filter by dt_days if provided
        # If user supplied dt_days (int or list[int]), check whether this file's
        # actual dt (in hours) matches any requested dt (days -> hours) within
        # the provided hourly tolerance. The check is performed in hours to
        # support hourly tolerances.
        dt_hours = int(abs((final_date - initial_date).total_seconds()) / 3600)
        dt_days_val = int(dt_hours // 24)

        if dt_days is not None:
            dt_filter_list = (
                tuple(dt_days) if isinstance(dt_days, list) else (int(dt_days),)
            )

        else:
            dt_filter_list = ()

        if dt_filter_list:
            matched = False
            for dt in dt_filter_list:
                target_hours = int(dt) * 24
                if abs(dt_hours - target_hours) <= int(dt_hours_tolerance):
                    matched = True
                    break
            if not matched:
                logger.debug(
                    f"Skipping {f.name}: dt_hours={dt_hours} not within any {dt_filter_list} ±{dt_hours_tolerance}h"
                )
                continue

        # append candidate using actual dt_hours and computed dt_days (int)
        candidates.append((f, final_date, initial_date, dt_hours, dt_days_val))

    total = len(candidates)
    done = 0
    loaded = 0
    if total == 0:
        logger.info("No candidate files found")
        return 0

    # Run parallel read/prepare tasks (safe for HDF5 / numpy since each job is a separate process)
    jobs = (
        delayed(_read_single_file)(
            str(f),
            final_date.isoformat(),
            initial_date.isoformat(),
            dt_hours,
            dt_days,
            invert_y,
        )
        for f, final_date, initial_date, dt_hours, dt_days in tqdm(
            candidates, desc="Reading DIC files", unit="file"
        )
    )
    results = Parallel(n_jobs=max_workers, prefer="processes")(jobs)

    # Sequentially store results into cache (single-threaded to keep cache consistent)
    for meta, payload in results:
        try:
            cache.store_dic_record(meta=meta, payload=payload)
            loaded += 1
        except Exception as e:
            logger.error(f"Failed storing record: {e}")
        done += 1
        if progress_callback:
            progress_callback(done, total)

    logger.info(f"Loaded {loaded} DIC files from disk (records={cache.num_records})")

    return loaded
