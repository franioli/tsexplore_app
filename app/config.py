from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

###=== SETTINGS SETUP ===###
CFG_PATH = Path(os.getenv("CONFIG_FILE", "config.yaml"))


# module-level cached instance (lazy)
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings


class Settings(BaseSettings):
    """Application settings with validation."""

    # Database connection
    use_database: bool = False
    db_host: str = "150.145.51.193"
    db_port: int = 5434
    db_name: str = "planpincieux"
    db_user: str = "postgres"
    db_password: str = ""

    # API connection
    use_api: bool = False
    api_host: str = "150.145.51.193"
    api_port: int = 8080

    # Data paths
    data_dir: Path = Path("./data/day_dic")
    file_search_pattern: str = "*.txt"

    # Filename template to extract intiial and final dates from filenames.
    # Template matches the file *stem* (filename without extension).
    # Use placeholders: {final:<strftime>} and {initial:<strftime>}.
    # Examples:
    # - day_dic_20230104-20230101.txt -> "day_dic_{final:%Y%m%d}-{initial:%Y%m%d}"
    filename_date_template: str | None = None

    # Backward compatible settings (deprecated): regex with 2 capture groups + a shared date_format
    # filename_pattern: str = r"day_dic_(\d{8})-(\d{8})"
    # date_format: str = "%Y%m%d"

    # Background image settings
    background_image: str | None = None  # Path to background image or None
    background_image_opacity: float = 1.0

    # Processing parameters
    dt_days: list[int] | int | None = None
    dt_hours_tolerance: int = 0
    invert_y: bool = False
    node_search_radius: float = 5.0

    # API defaults
    default_colorscale: str = "Reds"
    default_plot_type: str = "scatter"
    default_marker_size: int = 4
    default_marker_opacity: float = 0.7
    default_downsample_points: int = 5000

    # Results directory
    results_dir: Path = Path("./results")

    # Logging
    log_level: str = "INFO"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Admin token
    admin_reload_token: str | None = None

    # UI configuration
    serve_ui: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: defaults -> .env -> YAML -> env vars (env vars highest)
        # In pydantic-settings v2, earlier sources have higher priority.
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=str(CFG_PATH))
        return (
            init_settings,
            env_settings,  # highest
            yaml_source,  # overrides .env
            dotenv_settings,
            file_secret_settings,  # secrets dir, if used
        )


###=== LOGGING SETUP ===###


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    return logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


_logger: logging.Logger = setup_logger()
