# FastAPI Time Series Application

This project is a FastAPI web application that allows users to visualize velocity fields and time series data through interactive plotting features.


## Installation

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install the application

```bash
# Clone or navigate to the repository
cd tsexplore_app

# Create virtual environment and install dependencies
uv venv  .venv             # create .venv managed by uv
source .venv/bin/activate  # Activate the environment (On Windows: .venv\Scripts\activate)
uv sync --locked           # install dependencies
```

### Install in development mode

For development, install extra dependencies and omit `--locked` flag (keep editable installs so code changes reflect immediately):

```bash
uv sync --dev
```


## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` / `config.yaml` with your data paths and settings.

### Filename date extraction

For file-based loading, the recommended option is a single template that defines both the
location of the dates in the filename and their format:

- `filename_date_template: "day_dic_{final:%Y%m%d}-{initial:%Y%m%d}"`

The template is matched against the file stem (filename without extension). Use the
placeholders `{final:<strftime>}` and `{initial:<strftime>}`.

## Usage

Run the server:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open your browser at: `http://localhost:8000`

## Data Format

Velocity data files should be organized as CSV with columns: `x,y,u,v,nmad`
- Files named as: `YYYYMMDD_dic.csv` or similar
- Place in the directory specified in `.env`