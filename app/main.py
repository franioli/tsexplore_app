"""FastAPI application for velocity field visualization and time series analysis."""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cache import AppState, cache
from .config import get_logger, get_settings
from .routers import router as api_router
from .services import build_kdtree, get_data_provider

logger = get_logger()
settings = get_settings()
app_state = AppState()

# Global provider variable initialized in lifespan()
provider = None  # type: ignore

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


def get_loaded_dates() -> list[str]:
    """Return dates from cache if present, otherwise from provider."""
    # Use cached preloaded range (DB) if present
    if cache.all_data:
        return sorted(cache.all_data.keys())
    # Fallback to provider's available dates (reads DB or lists files)
    provider = get_data_provider()
    try:
        return provider.get_available_dates()
    except Exception:
        return []  # will display "No data" in UI; debug logs should explain failure


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: preload data on startup."""
    global provider
    logger.info("Starting application - initializing provider")
    try:
        # Initialize data provider (file or DB)
        provider = get_data_provider()
        logger.info(f"Using data provider: {provider.__class__.__name__}")

        # If not using DB backend preload all data and build spatial index
        if not settings.use_database:
            logger.info("Non-DB backend detected - preloading all data")
            all_data = provider.load_all()
            logger.info(f"Preloaded {len(all_data)} DIC records")

            tree, coords = build_kdtree(provider)
            logger.info(f"Built KDTree with {len(coords)} nodes")
        else:
            # Load only one sample day to display correctly the ui,

            logger.info(
                "DB backend detected - not preloading data at startup (use loader to preload a range)"
            )

        app_state.mark_ready()
        logger.info("Application ready")

    except Exception as e:
        logger.error(f"Failed during startup initialization: {e}", exc_info=True)
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


# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include API endpoints from the routers folder
app.include_router(api_router, prefix="/api")


# Serve HTML UI if enabled in settings
if settings.serve_ui:

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Serve the main HTML page (only when serve_ui is True)."""
        if not app_state.is_ready():
            return HTMLResponse(
                "<h1>Service starting up, please wait...</h1>", status_code=503
            )

        dates_raw = get_loaded_dates()
        if not dates_raw and settings.use_database:
            # DB backend but no dates known yet - UI should show load controls
            dates_iso = []
        else:
            dates_iso = [
                datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d") for d in dates_raw
            ]

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "dates_iso": dates_iso,
                "background_image": settings.background_image,
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
