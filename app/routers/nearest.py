from fastapi import APIRouter, HTTPException, Query

from ..config import get_logger, get_settings
from ..models.types import NearestNodeResponse
from ..services.spatial import nearest_node as find_nearest_node

logger = get_logger()
settings = get_settings()


router = APIRouter()


@router.get("/", response_model=NearestNodeResponse)
async def api_nearest(
    reference_date: str = Query(...),
    x: float = Query(...),
    y: float = Query(...),
    radius: float = Query(10.0, ge=0, le=1000),
):
    """Find nearest node."""
    logger.info(
        f"API /nearest - reference_date={reference_date}, x={x}, y={y}, radius={radius}"
    )
    try:
        node = find_nearest_node(
            x=x,
            y=y,
            reference_date=reference_date,
            radius=radius,
        )

        if node is None:
            logger.warning(f"No node found within radius {radius}")
            raise HTTPException(status_code=404, detail="No node within radius")

        logger.info(f"Found nearest node: ({node['x']:.2f}, {node['y']:.2f})")
        return NearestNodeResponse(
            node_id=int(node["node_id"]), x=float(node["x"]), y=float(node["y"])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding nearest node: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
