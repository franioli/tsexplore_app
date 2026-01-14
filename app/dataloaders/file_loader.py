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
from ..config import Settings, get_settings
from .data_provider import DataProvider

logger = logging.getLogger(__name__)


class FileDataProvider(DataProvider):
    """Load DIC data from text files on disk."""

    def __init__(
        self, cache: DataCache | None = None, settings: Settings | None = None
    ):
        """Initialize the file-backed provider."""

        # use provided settings or global
        self.settings = settings if settings is not None else get_settings()

        # allow cache injection; default to module-level global cache for backwards compat
        self._cache = cache if cache is not None else global_cache

        # Check for input files. Raise error if no valid files found.
        data_path = Path(self.settings.data_dir)
        files = list(data_path.glob(self.settings.file_search_pattern))
        if not files:
            raise FileNotFoundError(
                f"No files found in {data_path} matching pattern {self.settings.file_search_pattern}"
            )
        logger.debug(f"Found {len(files)} files in {data_path}")

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

        # Use glob to find files. The pattern might be just *.nc or *.h5
        files = data_path.glob(s.file_search_pattern)

        dates: set[str] = set()

        # == Prioritized check for NetCDF files first
        nc_files = [f for f in files if f.suffix == ".nc"]
        if nc_files:
            for f in nc_files:
                try:
                    nc_dates = _extract_dates_from_netcdf(f)
                    dates.update(nc_dates)
                except Exception as e:
                    logger.warning(f"Error reading dates from NetCDF {f.name}: {e}")

            return sorted(dates)

        # == Check other files

        # date pattern in filenames
        filename_date_template = getattr(s, "filename_date_template", None)

        other_files = [f for f in files if f.suffix != ".nc"]
        for f in other_files:
            try:
                final_date, _initial_date = _parse_filename(
                    f,
                    filename_date_template=filename_date_template,
                )
                dates.add(final_date.strftime("%Y-%m-%d"))
            except Exception as e:
                logger.warning(f"Skipping file {f.name}: {e}")
                continue

        return sorted(dates)

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
            max_workers=self.settings.max_workers,
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

            # Filter by dt_days if requested
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


def _extract_dates_from_netcdf(
    file_path: Path, date_var_name: str = "date2"
) -> list[str]:
    """Peek at a NetCDF file to get the list of final dates (slave dates)."""
    import xarray as xr

    dates = []
    try:
        # Use decode_times=True to get datetime objects immediately
        with xr.open_dataset(file_path, decode_times=True) as ds:
            # We assume reference date is 'date2' (slave) as per stack script
            if date_var_name in ds.coords or date_var_name in ds.data_vars:
                date2_vals = ds[date_var_name].values
                # Convert numpy datetime64 to strings
                dates = [
                    np.datetime_as_string(d, unit="D")
                    for d in np.atleast_1d(date2_vals)
                ]
    except Exception as e:
        logger.error(f"Failed to read dates from {file_path}: {e}")
    return dates


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
        for (ph, fmt), val in zip(placeholders, groups, strict=False):
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

    # helper to read a dataset into preallocated array (if present)
    def _read_ds(group, name, dtype=np.float32, shape=None):
        if name not in group:
            return np.zeros(shape or (n_nodes,), dtype=dtype)
        ds = group[name]
        out = np.empty(ds.shape, dtype=ds.dtype)
        ds.read_direct(out)
        return out

    logger.debug(f"Reading HDF5 file: {file_path}")

    with h5py.File(file_path, "r") as f:
        # pick ensemble group or first pair group
        if "ensemble" in f:
            logger.debug("Using 'ensemble' group")
            g = f["ensemble"]
        else:
            logger.debug("Ensemble group not found, using first data group")
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

        nodes = _read_ds(g, "nodes", dtype=np.float32)
        if nodes.ndim == 2 and nodes.shape[1] >= 2:
            x = nodes[:, 0].astype(np.float32)
            y = nodes[:, 1].astype(np.float32)
        else:
            x = _read_ds(g, "x", dtype=np.float32, shape=(n_nodes,))
            y = _read_ds(g, "y", dtype=np.float32, shape=(n_nodes,))

        dx = _read_ds(g, "dx", dtype=np.float32, shape=(n_nodes,))
        dy = _read_ds(g, "dy", dtype=np.float32, shape=(n_nodes,))

        # ensure 1-D and length-consistent
        x = np.asarray(x, dtype=np.float32).ravel()[:n_nodes]
        y = np.asarray(y, dtype=np.float32).ravel()[:n_nodes]
        dx = np.asarray(dx, dtype=np.float32).ravel()[:n_nodes]
        dy = np.asarray(dy, dtype=np.float32).ravel()[:n_nodes]

        if invert_y:
            y = -y
            dy = -dy

        magnitude = np.hypot(dx, dy)

        # read additional fields if present
        additionals: dict[str, np.ndarray] = {}
        if "mad" in g:
            mad = _read_ds(g, "mad", dtype=np.float32, shape=(n_nodes,))
            additionals["ensemble_mad"] = mad
        elif "ensemble_mad" in g:
            mad = _read_ds(g, "ensemble_mad", dtype=np.float32, shape=(n_nodes,))
            additionals["ensemble_mad"] = mad
        else:
            additionals["ensemble_mad"] = np.full((n_nodes,), np.nan, dtype=np.float32)
            logger.debug("No ensemble_mad/mad found, filling with NaNs")

        if "corr" in g:
            corr = _read_ds(g, "corr", dtype=np.float32, shape=(n_nodes,))
            additionals["corr_score"] = corr

        # Manage additional fields
        for k, v in additionals.items():
            v_arr = np.asarray(v, dtype=np.float32).ravel()[:n_nodes]
            additionals[k] = v_arr

        logger.debug(f"Read {n_nodes} nodes from HDF5")
        logger.debug(additionals.keys())
        logger.debug("  " + str(additionals["ensemble_mad"]))
        logger.debug("-----")

    data = {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
    }
    data.update(additionals)

    return data


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


def _load_netcdf_cube_into_cache(
    file_path: Path,
    cache: DataCache,
    invert_y: bool = False,
    dt_days_filter: int | list[int] | None = None,
    dt_hours_tolerance: int = 0,
    reference_date_start: datetime | None = None,
    reference_date_end: datetime | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
):
    """Load records from a single NetCDF cube into the cache.

    The NetCDF file is expected to have 'vx', 'vy', 'mad' variables and 'date1', 'date2', 'dt_days'.
    It flattens the grid for each timestamp and ignores NaN entries to match sparse cache format.
    """
    import xarray as xr

    logger.info(f"Loading NetCDF cube: {file_path}")

    # We load the dataset. Chunks=None to load into memory or use Dask if installed,
    # but here we iterate efficiently.
    with xr.open_dataset(file_path, decode_times=True) as ds:
        # Check global attributes for mode
        mode = ds.attrs.get("obs_mode", "velocity")  # 'velocity' or 'displacement'

        # Grid coordinates
        xs = ds.x.values
        ys = ds.y.values
        X, Y = np.meshgrid(xs, ys)

        # Flatten grid arrays once
        X_flat = X.ravel().astype(np.float32)
        Y_flat = Y.ravel().astype(np.float32)

        if invert_y:
            Y_flat = -Y_flat

        # Identify time dimension (usually 'mid_date')
        # However, date1/date2 might be data_vars or coords with same dim as mid_date
        # We iterate over the 'mid_date' dimension size
        if "mid_date" not in ds.dims:
            logger.error("NetCDF file missing 'mid_date' dimension.")
            return

        num_steps = ds.dims["mid_date"]

        # Pre-load time variables to avoid random access overhead
        # Use .values to get numpy arrays (datetime64)
        dates1 = ds["date1"].values
        dates2 = ds["date2"].values
        dts_days = ds["dt_days"].values

        processed_count = 0

        for i in tqdm(range(num_steps), desc="Reading NetCDF cube", unit="slice"):
            # Extract metadata
            # Convert datetime64[ns] to datetime object
            date1_np = dates1[i]
            date2_np = dates2[i]

            # Safe conversion
            ts_start = (
                date1_np - np.datetime64("1970-01-01T00:00:00Z")
            ) / np.timedelta64(1, "s")
            ts_end = (
                date2_np - np.datetime64("1970-01-01T00:00:00Z")
            ) / np.timedelta64(1, "s")
            initial_date_ = datetime.utcfromtimestamp(ts_start)
            final_date_ = datetime.utcfromtimestamp(ts_end)

            # --- Filters ---
            if (
                reference_date_start
                and final_date_.date() < reference_date_start.date()
            ):
                continue
            if reference_date_end and final_date_.date() > reference_date_end.date():
                continue

            dt_days_val = float(dts_days[i])
            dt_hours = dt_days_val * 24.0

            if dt_days_filter is not None:
                filter_list = (
                    tuple(dt_days_filter)
                    if isinstance(dt_days_filter, list)
                    else (int(dt_days_filter),)
                )
                matched = False
                for ft in filter_list:
                    target_hours = ft * 24.0
                    if abs(dt_hours - target_hours) <= dt_hours_tolerance:
                        matched = True
                        break
                if not matched:
                    continue

            # --- Extract Data Slice ---
            # Slice the i-th step. .values loads it.
            # Variables are likely (mid_date, y, x)
            vx_slice = ds["vx"].isel(mid_date=i).values.ravel()
            vy_slice = ds["vy"].isel(mid_date=i).values.ravel()

            if "mad" in ds:
                mad_slice = ds["mad"].isel(mid_date=i).values.ravel()
            else:
                mad_slice = np.full_like(vx_slice, np.nan)

            # Filter NaNs (valid mask)
            # We assume vx and vy have NaNs in the same places (outside image overlap)
            valid_mask = ~np.isnan(vx_slice)

            if not np.any(valid_mask):
                continue

            x_valid = X_flat[valid_mask]
            y_valid = Y_flat[valid_mask]
            vx_valid = vx_slice[valid_mask]
            vy_valid = vy_slice[valid_mask]
            mad_valid = mad_slice[valid_mask] if mad_slice is not None else None

            if invert_y:
                vy_valid = -vy_valid

            # Compute Derived Fields based on mode
            if mode == "velocity":
                # Data is velocity (px/d). Compute displacement.
                # If dt is very small, this might be unstable, but dt usually >= 1 day
                u_vel = vx_valid
                v_vel = vy_valid
                dx = u_vel * dt_days_val
                dy = v_vel * dt_days_val
            else:
                # Data is displacement (px). Compute velocity.
                dx = vx_valid
                dy = vy_valid
                if dt_days_val != 0:
                    scale = 1.0 / dt_days_val
                    u_vel = dx * scale
                    v_vel = dy * scale
                else:
                    u_vel = dx
                    v_vel = dy

            disp_mag = np.hypot(dx, dy)
            V_vel = np.hypot(u_vel, v_vel)

            # --- Store Record ---
            meta = {
                "initial_date": initial_date_,
                "final_date": final_date_,
                "reference_date": final_date_,
                "dt_hours": int(dt_hours),
                "dt_days": int(dt_days_val),
            }

            payload = {
                "x": x_valid,
                "y": y_valid,
                "dx": dx.astype(np.float32),
                "dy": dy.astype(np.float32),
                "disp_mag": disp_mag.astype(np.float32),
                "u": u_vel.astype(np.float32),
                "v": v_vel.astype(np.float32),
                "V": V_vel.astype(np.float32),
                "ensemble_mad": mad_valid.astype(np.float32)
                if mad_valid is not None
                else None,
                "dt_hours": int(dt_hours),
                "dt_days": int(dt_days_val),
                "initial_date": initial_date_.strftime("%Y-%m-%d"),
                "final_date": final_date_.strftime("%Y-%m-%d"),
                "reference_date": final_date_.strftime("%Y-%m-%d"),
            }

            cache.store_dic_record(meta=meta, payload=payload)
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, num_steps)

    logger.info(f"Loaded {processed_count} records from NetCDF cube.")
    return processed_count


def _load_data_to_cache(
    *,
    data_dir: str | Path,
    file_search_pattern: str = "*.nc",
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

    if not files:
        logger.warning("No files found.")
        return 0

    # Check for NetCDF cubes explicitly first
    nc_files = sorted(data_path.glob("*.nc"))
    loaded_count = 0
    if nc_files:
        logger.info(f"Found {len(nc_files)} NetCDF cube(s). Using NetCDF only.")
        loaded_count = 0
        for f in nc_files:
            try:
                n = _load_netcdf_cube_into_cache(
                    file_path=f,
                    cache=cache,
                    invert_y=invert_y,
                    dt_days_filter=dt_days,
                    dt_hours_tolerance=dt_hours_tolerance,
                    reference_date_start=reference_date_start,
                    reference_date_end=reference_date_end,
                    progress_callback=progress_callback,
                )
                if n:
                    loaded_count += n
                else:
                    logger.info(f"No records loaded from NetCDF cube {f.name}.")
            except Exception as e:
                logger.error(f"Failed to load NetCDF cube {f.name}: {e}")

        return loaded_count

    # If no NC files, search for standard files using the provided pattern
    files = sorted(data_path.glob(file_search_pattern))

    # Filter out .nc files from this list just in case pattern matches them (e.g. *.*)
    standard_files = [f for f in files if f.suffix != ".nc"]

    if not standard_files:
        logger.warning(f"No files found matching {file_search_pattern} in {data_path}")
        return 0
    logger.info(f"Found {len(standard_files)} DIC files.")

    # Pre-filter files by parsing filenames and applying date / dt filters
    candidates: list[tuple[Path, datetime, datetime, int, int]] = []
    for f in standard_files:
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
