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

    # Data paths
    data_dir: Path = Path("./data/day_dic")
    file_pattern: str = "*.txt"  # glob pattern to match data files # TODO: avoid ambiguity with filename_pattern
    filename_pattern: str = r"day_dic_(\d{8})-(\d{8})"  # regex to extract date range
    background_image: str = ""

    # Processing parameters
    dt_days_preferred: int = 3
    invert_y: bool = False
    date_format: str = "%Y%m%d"
    search_radius: float = 5.0

    # API defaults
    default_colorscale: str = "Reds"
    default_plot_type: str = "scatter"
    default_marker_size: int = 5
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


# Global settings instance
settings = Settings()
