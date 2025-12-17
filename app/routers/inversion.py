import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from ..cache import cache
from ..config import get_logger, get_settings
from ..services.data_provider import get_data_provider
from ..services.inversion import invert_node, load_dic_data
from ..services.spatial import build_kdtree

logger = get_logger()
settings = get_settings()

router = APIRouter()


@router.post(
    "/inversion/run",
    summary="Run time-series inversion (node series provided)",
    tags=["inversion"],
)
async def run_node_inversion(
    node_x: float = Body(..., description="Node X coordinate"),
    node_y: float = Body(..., description="Node Y coordinate"),
    weight_method: str | None = Query("variable", description="Weighting method"),
    regularization_method: str = Query(
        "laplacian", description="Regularization method"
    ),
    lambda_scaling: float = Query(1.0, description="Scaling for lambda"),
    iterates: int = Query(10, ge=1, description="Number of inversion iterations"),
):
    """
    Run time-series inversion on provided time-series data.
    """
    # Fail early if no data loaded
    if not cache.is_loaded():
        logger.warning("Nearest requested but no data has been loaded")
        raise HTTPException(
            status_code=400,
            detail="No data loaded yet. Press 'Load' before requesting nearest node.",
        )

    logger.info(f"Performing inversion for node at ({node_x}, {node_y})")

    try:
        logger.info("Fetching time series data")
        provider = get_data_provider()

        # Do not group by dt: we need the full list of observations for inversion
        ts_groups = provider.extract_node_timeseries(
            node_x=node_x, node_y=node_y, dt_days=None, group_by_dt=False
        )
        if not ts_groups:
            raise ValueError("No time-series data available for the requested node")

        # extract the single (ungrouped) timeseries group
        # provider returns {0: {...}} when group_by_dt=False, but handle generic case
        main_group = ts_groups.get(0) or ts_groups[next(iter(ts_groups.keys()))]
        n_obs = main_group["dx"].shape[0]
        if n_obs < 1:
            raise ValueError("Not enough observations for inversion")

        # Build EW/NS series (displacements) and timestamp array (n_obs, 2)
        ew = np.asarray(main_group["dx"], dtype=np.float32)
        ns = np.asarray(main_group["dy"], dtype=np.float32)

        final_dates = np.asarray(main_group["final_dates"], dtype="datetime64[D]")
        initial_dates = np.asarray(main_group["initial_dates"], dtype="datetime64[D]")

        if final_dates.shape[0] != n_obs or initial_dates.shape[0] != n_obs:
            raise ValueError("Inconsistent date arrays in timeseries data")

        timestamp = np.empty((n_obs, 2), dtype="datetime64[D]")
        timestamp[:, 0] = final_dates  # master / final
        timestamp[:, 1] = initial_dates  # slave / initial

        # compute deltat in days (float32)
        # deltat = np.abs(
        #     (timestamp[:, 0] - timestamp[:, 1]).astype("timedelta64[D]").astype(float)
        # ).astype("float32")

        # optional per-observation ensemble MAD weights
        ens_mad = main_group.get("ensamble_mad", None)
        weight_var = (
            np.asarray(ens_mad, dtype=np.float32) if ens_mad is not None else None
        )

        logger.info("Data prepared (%d observations). Running inversion...", n_obs)
        res = invert_node(
            ew_series=ew,
            ns_series=ns,
            timestamp=timestamp,
            weight_method=weight_method,
            weight_variable=weight_var,
            regularization_method=regularization_method,
            lambda_scaling=lambda_scaling,
            iterates=iterates,
        )
        if not res:
            raise RuntimeError("invert_node returned no result")

        # normalize numpy arrays to plain JSON-friendly lists
        try:
            ew_hat = np.asarray(res["EW_hat"]).tolist()
            ns_hat = np.asarray(res["NS_hat"]).tolist()
            # Time_hat may be numpy datetime; extract day-string per sample
            time_hat = np.asarray(res["Time_hat"])
            try:
                dates_arr = time_hat[:, 1]
            except Exception:
                dates_arr = time_hat
            dates = [str(d) for d in np.datetime_as_string(dates_arr, unit="D")]
            V_hat = (np.sqrt(np.array(ew_hat) ** 2 + np.array(ns_hat) ** 2)).tolist()
        except Exception:
            ew_hat = res.get("EW_hat")
            ns_hat = res.get("NS_hat")
            dates = res.get("Time_hat")
            V_hat = None

        logger.info("Inversion completed.")

        return JSONResponse(
            {
                "status": "ok",
                "node_inversion": {
                    "EW_hat": ew_hat,
                    "NS_hat": ns_hat,
                    "dates": dates,
                    "V_hat": V_hat,
                },
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        logger.warning("Inversion not supported: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        logger.exception("node inversion failed")
        raise HTTPException(status_code=500, detail="inversion failure")


@router.post(
    "/inversion/run-from-files",
    summary="Run node inversion by extracting series from files (local data)",
    tags=["inversion"],
)
async def run_node_inversion_from_files(
    node_x: float = Body(..., description="Node X coordinate"),
    node_y: float = Body(..., description="Node Y coordinate"),
    iterates: int = Body(10, ge=1, description="Number of inversion iterations"),
    regularization_method: str = Body("laplacian", description="Regularization method"),
    lambda_scaling: float = Body(1.0, description="Scaling for lambda"),
    weight_method: str = Body("residuals", description="Weighting method"),
):
    """
    Run node-level inversion by extracting ew/ns time-series for (node_x,node_y)
    from local files and running the inversion.
    """
    try:
        logger.info(f"Performing inversion for node at ({node_x}, {node_y})")

        logger.info("Loading data...")
        provider = get_data_provider()
        tree, coords = build_kdtree(provider)
        dist, idx = tree.query([node_x, node_y], k=1)
        if not np.isfinite(dist):
            raise ValueError("node not found in KDTree")
        dic = load_dic_data(
            settings.data_dir,
            file_pattern=settings.file_search_pattern,
        )
        ew_series = dic["ew"][:, idx]
        ns_series = dic["ns"][:, idx]
        ens_mad = dic["weight"][:, idx] if "weight" in dic else None
        timestamp = dic["timestamp"]

        # Ensure numpy arrays
        ew = np.asarray(ew_series)
        ns = np.asarray(ns_series)
        ts = np.asarray(timestamp)

        if ew.shape[0] != ns.shape[0] or ew.shape[0] != ts.shape[0]:
            raise ValueError("EW/NS/Timestamp arrays must have same length")

        logger.info("Data loaded. Running inversion...")
        res = invert_node(
            ew_series=ew,
            ns_series=ns,
            timestamp=ts,
            weight_method=weight_method,
            weight_variable=ens_mad,
            regularization_method=regularization_method,
            lambda_scaling=lambda_scaling,
            iterates=iterates,
        )

        # normalize numpy arrays to plain JSON-friendly lists
        try:
            ew_hat = np.asarray(res["EW_hat"]).tolist()
            ns_hat = np.asarray(res["NS_hat"]).tolist()
            # Time_hat may be numpy datetime; extract day-string per sample
            time_hat = np.asarray(res["Time_hat"])
            try:
                dates_arr = time_hat[:, 1]
            except Exception:
                dates_arr = time_hat
            dates = [str(d) for d in np.datetime_as_string(dates_arr, unit="D")]
            V_hat = (np.sqrt(np.array(ew_hat) ** 2 + np.array(ns_hat) ** 2)).tolist()
        except Exception:
            ew_hat = res.get("EW_hat")
            ns_hat = res.get("NS_hat")
            dates = res.get("Time_hat")
            V_hat = None

        logger.info("Inversion completed.")

        return JSONResponse(
            {
                "status": "ok",
                "node_inversion": {
                    "EW_hat": ew_hat,
                    "NS_hat": ns_hat,
                    "dates": dates,
                    "V_hat": V_hat,
                },
            }
        )

    except HTTPException:
        raise
    except NotImplementedError as e:
        logger.warning("Inversion not supported: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("node inversion (from files) failed")
        raise HTTPException(status_code=500, detail=f"inversion failure: {e}") from e
