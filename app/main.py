"""FastAPI application for velocity field visualization and time series analysis."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cache import AppState, cache
from .config import get_logger, get_settings, reload_settings
from .routers import router as api_router
from .services import get_data_provider

logger = get_logger()
app_state = AppState()

# Global variables initialized in lifespan()
settings = None
provider = None  # type: ignore
available_dates = []  # type: ignore
startup_error: str | None = None


# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: do NOT preload data at startup (lazy loading)."""
    global settings, provider, available_dates, startup_error
    logger.info("Starting application - initializing provider (lazy loading enabled)")
    # Load settings (or reload if they are already present)
    settings = get_settings() if settings is None else reload_settings()
    logger.info("Settings loaded")

    try:
        provider = get_data_provider()
        if provider is None:
            raise RuntimeError("No valid data provider could be initialized")
        logger.info(f"Using data provider: {provider.__class__.__name__}")

        available_dates = provider.get_available_dates()
        if not available_dates:
            raise RuntimeError(
                "No available dates found in data provider. Check data source or configuration file."
            )
        logger.debug(f"Available dates: {available_dates}")

        # Default behavior: do not load any data until user presses "Load"
        logger.debug("Skipping preload at startup; waiting for user-triggered load")

        app_state.mark_ready()
        logger.info("Application ready")

    except Exception as e:
        # record a short, safe message for the frontend and keep process running
        startup_error = f"{type(e).__name__}: {e}"
        logger.error(
            "Failed during startup initialization: %s", startup_error, exc_info=True
        )
        # Do not re-raise so the server can start and the frontend can show the error

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


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if not app_state.is_ready():
        # show startup error if present, otherwise generic starting message
        if startup_error:
            return HTMLResponse(
                f"<h1>Startup error</h1><pre>{startup_error}</pre>",
                status_code=500,
            )

        return HTMLResponse(
            "<h1>Service starting up, please wait...</h1>", status_code=503
        )
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
