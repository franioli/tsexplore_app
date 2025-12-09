FROM ghcr.io/astral-sh/uv:0.9.16-trixie-slim

WORKDIR /app

# copy dependency metadata first to leverage caching
COPY pyproject.toml uv.lock* /app/

# install dependencies (no project) with uv, cache mount speeds builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# copy project sources
COPY . /app

# install project (editable) into .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# make the venv available on PATH
ENV PATH="/app/.venv/bin:${PATH}"
ENV UV_COMPILE_BYTECODE=1

EXPOSE 8000

# Default run — in development you can override to add --reload or mount code
CMD ["uv", "run", "uvicorn src.app:app", "--host", "0.0.0.0", "--port", "8000"]