# FastAPI Time Series Application

This project is a FastAPI web application that allows users to visualize velocity fields and time series data through interactive plotting features. Users can select specific dates and points on the image to generate velocity plots and time series graphs.


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

Velocity data files should be organized as CSV with columns: `x,y,u,v,V,nmad`
- Files named as: `YYYYMMDD_dic.csv` or similar
- Place in the directory specified in `.env`
## Project Structure

```
fastapi-timeseries-app
├── src
│   ├── main.py                # Entry point of the FastAPI application
│   ├── api
│   │   ├── __init__.py        # API module initializer
│   │   └── routes.py          # API routes for fetching data and handling requests
│   ├── core
│   │   ├── __init__.py        # Core module initializer
│   │   ├── config.py          # Configuration settings for the application
│   │   └── data_loader.py      # Functions to read velocity fields from day-dic files
│   ├── models
│   │   ├── __init__.py        # Models module initializer
│   │   └── schemas.py         # Data schemas for request and response validation
│   ├── services
│   │   ├── __init__.py        # Services module initializer
│   │   ├── plotting.py         # Functions for generating plots
│   │   └── timeseries.py       # Functions for handling time series data
│   └── utils
│       ├── __init__.py        # Utils module initializer
│       └── helpers.py         # Utility functions for various tasks
├── static
│   ├── css
│   │   └── style.css          # CSS styles for the web application
│   ├── js
│   │   └── app.js             # JavaScript code for user interactions
│   └── images                 # Directory for static images
├── templates
│   └── index.html             # Main HTML template for the web application
├── data
│   └── temp_data              # Directory for day-dic files with velocity data
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables for configuration
├── .gitignore                  # Files and directories to ignore by Git
└── README.md                   # Documentation for the project
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.