from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VelocityMapRequest(BaseModel):
    """Request model for velocity map endpoint."""

    date: str = Field(..., description="Date in YYYYMMDD format")
    use_velocity: bool = Field(True, description="Use velocity instead of displacement")
    cmin: float | None = Field(None, ge=0, description="Minimum color scale value")
    cmax: float | None = Field(None, ge=0, description="Maximum color scale value")
    colorscale: Literal["Reds", "Viridis", "Plasma", "Blues", "RdYlBu", "Turbo"] = (
        "Reds"
    )
    plot_type: Literal["scatter", "raster"] = "scatter"
    marker_size: int = Field(6, ge=1, le=20, description="Marker size in pixels")
    marker_opacity: float = Field(0.7, ge=0, le=1, description="Marker opacity")
    downsample_points: int = Field(5000, ge=100, description="Max points to display")
    selected_x: float | None = None
    selected_y: float | None = None

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError as e:
            raise ValueError("Date must be in YYYY-MM-DD format") from e


class PlotlyFigure(BaseModel):
    """Simple wrapper for Plotly figure dictionaries."""

    fig: dict = Field(..., description="Plotly figure as dict")


class VelocityMapResponse(BaseModel):
    """Response model for /map endpoints."""

    ok: bool = True
    date: str
    use_velocity: bool
    n_points: int | None = None
    figure: dict | None = None  # plotly fig dict
    meta: dict | None = None


class TimeSeriesRequest(BaseModel):
    """Request model for time series endpoint."""

    node_x: float = Field(..., description="X coordinate of node")
    node_y: float = Field(..., description="Y coordinate of node")
    use_velocity: bool = True
    components: str = Field("V", description="Comma-separated: u,v,V")
    marker_mode: Literal["lines+markers", "lines", "markers"] = "lines+markers"
    xmin_date: str | None = None
    xmax_date: str | None = None
    ymin: float | None = None
    ymax: float | None = None
    show_error_band: bool = False
    ts_inversion: bool = False

    @field_validator("components")
    @classmethod
    def validate_components(cls, v: str) -> str:
        """Validate components are valid."""
        valid = {"u", "v", "V"}
        parts = [c.strip() for c in v.split(",")]
        if not all(c in valid for c in parts):
            raise ValueError(f"Components must be from: {valid}")
        return v


class TimeSeriesPoint(BaseModel):
    """Single point in a timeseries."""

    date: str  # YYYYMMDD
    u: float | None = None
    v: float | None = None
    V: float | None = None
    u_std: float | None = None
    v_std: float | None = None
    V_std: float | None = None


class InversionResult(BaseModel):
    """Compact inversion result bundle for a node."""

    dates_inv: list[str]  # YYYYMMDD
    EW_hat: list[float]
    NS_hat: list[float]
    V_inv: list[float]


class TimeSeriesResponse(BaseModel):
    """Response model for /timeseries endpoints."""

    node_x: float
    node_y: float
    ok: bool = True
    dates: list[str]  # original dates (YYYYMMDD)
    series: list[TimeSeriesPoint]  # raw series values per-date
    figure: dict | None = None  # optional plotly dict
    inversion: InversionResult | None = None
    meta: dict | None = None

    @field_validator("dates")
    @classmethod
    def validate_dates(cls, v: list[str]) -> list[str]:
        """Ensure dates are YYYYMMDD formatted strings."""
        for d in v:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Dates must be list of strings in YYYY-MM-DD format")
        return v


class NearestNodeRequest(BaseModel):
    """Request model for nearest node endpoint."""

    date: str = Field(..., description="Date in YYYYMMDD format")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    radius: float = Field(10.0, ge=0, le=1000, description="Search radius in pixels")

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y%m%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYYMMDD format")


class NearestNodeResponse(BaseModel):
    """Response model for nearest node endpoint."""

    node_id: int
    x: float
    y: float


class InversionConfig(BaseModel):
    """Configuration for time series inversion."""

    weight_method: Literal["residuals", "uniform", "variable"] = "residuals"
    regularization_method: Literal["laplacian", "none"] = "laplacian"
    lambda_scaling: float | Literal["auto_std", "auto_mad"] = 1.0
    iterates: int = Field(10, ge=1, le=100, description="Max iterations")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str
    data_loaded: bool
    kdtree_built: bool
    num_dates: int | None = None


class ErrorResponse(BaseModel):
    error: str
