import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )

    serve_ui: bool = True  # if False FastAPI will not register the HTML UI routes

    # Data paths
    data_dir: Path = Path("./data/day_dic")
    file_pattern: str = "*.txt"  # glob pattern to match data files # TODO: avoid ambiguity with filename_pattern
    filename_pattern: str = r"day_dic_(\d{8})-(\d{8})"  # regex to extract date range
    background_image: str = ""
    background_image_opacity: float = 1.0

    # Processing parameters
    dt_days_preferred: int = 3
    invert_y: bool = False
    date_format: str = "%Y%m%d"
    search_radius: float = 5.0

    # API defaults
    default_colorscale: str = "Reds"
    default_plot_type: str = "scatter"
    default_marker_size: int = 4
    default_marker_opacity: float = 0.7
    default_downsample_points: int = 5000

    # Results and search
    results_dir: Path = Path("./results")

    # Logging
    log_level: str = "INFO"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Admin token to protect reload endpoint (optional)
    admin_reload_token: str | None = None


# module-level cached instance (lazy)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the live settings instance. Lazily instantiate on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Recreate the Settings instance (reads env file / env vars again)."""
    global _settings
    _settings = Settings()
    return _settings


# logger setup


def setup_logger() -> logging.Logger:
    """Setup the module-level logger."""
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # ensure reconfiguration in case of multiple calls
    )
    return logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    """Return the module-level logger."""
    global _logger
    if _logger is None:
        _logger = setup_logger()

    return _logger


_logger: logging.Logger = setup_logger()
