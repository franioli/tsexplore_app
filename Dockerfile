FROM ghcr.io/astral-sh/uv:0.9.16-trixie-slim

ENV UV_COMPILE_BYTECODE=1
WORKDIR /app

# Create a persistent venv outside /app so bind-mounting /app won't hide it
RUN uv venv /opt/.venv

# tell uv which project environment to use
ENV UV_PROJECT_ENVIRONMENT=/opt/.venv
ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="/opt/.venv/bin:${PATH}"

# copy dependency metadata install dependencies (no project) to leverage layer caching if no dependencies changed
COPY pyproject.toml uv.lock* /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# copy app sources into /app and install the project into the created venv (editable by default). 
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

EXPOSE 8000

# default run command — use `--reload` during development by overriding CMD in docker run/compose
CMD ["uv","--active", "run", "uvicorn src.app:app", "--host", "0.0.0.0", "--port", "8000"]