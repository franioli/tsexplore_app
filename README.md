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
cd ts_inversion_app

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development (includes testing tools)
uv pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your data paths and settings.

## Usage

Run the server:

```bash
uv run uvicorn src.main:app --reload --port 8000
```

Open your browser at: `http://localhost:8000`

## Data Format

Velocity data files should be organized as CSV with columns: `x,y,u,v,nmad`
- Files named as: `YYYYMMDD_dic.csv` or similar
- Place in the directory specified in `.env`