"""FastAPI application for velocity field visualization and time series analysis."""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cache import get_loaded_dates
from .config import get_logger, get_settings
from .routers import router as api_router
from .services.data import (
    build_global_kdtree,
    preload_all_data,
)

settings = get_settings()
logger = get_logger()


# Global app state
class AppState:
    """Centralized application state."""

    def __init__(self):
        self.data_loaded = False
        self.kdtree_built = False

    def is_ready(self) -> bool:
        """Check if app is ready to serve requests."""
        return self.data_loaded and self.kdtree_built


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: preload data on startup."""
    logger.info("Starting application - preloading data...")
    try:
        preload_all_data(
            data_dir=settings.data_dir,
            file_pattern=settings.file_pattern,
            filename_pattern=settings.filename_pattern,
            dt_days_preferred=settings.dt_days_preferred,
        )
        app_state.data_loaded = True
        logger.info("Data preloaded successfully")

        build_global_kdtree(
            data_dir=settings.data_dir,
            file_pattern=settings.file_pattern,
            filename_pattern=settings.filename_pattern,
        )
        app_state.kdtree_built = True
        logger.info("KDTree built successfully")

    except Exception as e:
        logger.error(f"Failed to preload data: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down application")


# Initialize FastAPI app
app = FastAPI(
    title="Velocity Field Time Series Viewer",
    description="Interactive visualization of velocity fields and time series data",
    version="0.1.0",
    lifespan=lifespan,
)

# Include API endpoints from the routers folder
app.include_router(api_router, prefix="/api")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# ===== HTML Endpoints =====
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page."""
    if not app_state.is_ready():
        return HTMLResponse(
            content="<h1>Service starting up, please wait...</h1>", status_code=503
        )

    dates_raw = get_loaded_dates()  # YYYYMMDD format

    # Convert to display format (DD/MM/YYYY) for user
    dates_display = [
        datetime.strptime(d, "%Y%m%d").strftime("%d/%m/%Y") for d in dates_raw
    ]

    # Convert dates to iso format for HTML5 date inputs (YYYY-MM-DD)
    dates_display_iso = [
        datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates_raw
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "dates_raw": dates_raw,
            "dates_display": dates_display,
            "dates_iso": dates_display_iso,
            "background_image": settings.background_image,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
