from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from ..config import get_logger, get_settings
from ..services.inversion import run_ts_inversion

logger = get_logger()
settings = get_settings()

router = APIRouter()


@router.post("/inversion/run", summary="Run time-series inversion", tags=["inversion"])
async def run_inversion(params: dict[str, Any] = Body(...)):
    """
    Run time-series inversion for the corpus or for a node.
    Accept keys: iterates, weight_method, weight_variable_data, regularization_method, lambda_scaling
    """
    try:
        result = run_ts_inversion(
            day_dic_dir=settings.data_dir,
            iterates=int(params.get("iterates", 10)),
            weight_method=params.get("weight_method", "residuals"),
            weight_variable_data=params.get("weight_variable_data"),
            regularization_method=params.get("regularization_method", "laplacian"),
            lambda_scaling=params.get("lambda_scaling", 1.0),
        )
        return JSONResponse(
            {"status": "ok", "result_shape": {"EW_hat": result["EW_hat"].shape}}
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.error("run_ts_inversion failed", exc_info=True)
        raise HTTPException(status_code=500, detail="inversion failure")
