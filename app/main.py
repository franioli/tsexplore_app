"""FastAPI application for velocity field visualization and time series analysis."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cache import AppState, cache
from .config import get_logger, get_settings
from .routers import router as api_router
from .services import get_data_provider

logger = get_logger()
settings = get_settings()
app_state = AppState()

# Global provider variable initialized in lifespan()
provider = None  # type: ignore

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: do NOT preload data at startup (lazy loading)."""
    global provider
    logger.info("Starting application - initializing provider (lazy loading enabled)")
    try:
        provider = get_data_provider()
        logger.info(f"Using data provider: {provider.__class__.__name__}")

        # Default behavior: do not load any data until user presses "Load"
        logger.debug("Skipping preload at startup; waiting for user-triggered load")

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
        if not app_state.is_ready():
            return HTMLResponse(
                "<h1>Service starting up, please wait...</h1>", status_code=503
            )

        # Always show *available* dates for the LOAD pickers (lazy, no data load)
        available_dates = provider.get_available_dates() if provider is not None else []

        # Show *loaded* dates for the main navigation/plots (depends on cache)
        loaded_dates = cache.get_available_dates() if cache.is_loaded() else []

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "available_dates_iso": available_dates,
                "loaded_dates_iso": loaded_dates,
                "use_database": settings.use_database,
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
