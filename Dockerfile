FROM python:3.13-slim-bookworm AS python_base

# Install system-level shared libs needed by compiled deps (e.g. rasterio -> libexpat)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends libexpat1

# Create a dedicated user to run all tasks (as non-root)
ARG UID=99
RUN useradd -u $UID -ms /bin/bash lulc
USER lulc

ENV HOME=/home/lulc
ENV WD=$HOME/package
WORKDIR $WD

# Build stage: having a build stage means temp/cache files from the build aren't persisted in the final image
FROM python_base AS builder

ENV UV_HOME="~/.cache/uv"
ENV PATH="$UV_HOME/bin:$PATH"

# Install uv in an isolated venv to avoid conflicts with the package venv
RUN --mount=type=cache,uid=$UID,target=$HOME/.cache/pip \
    python3 -m venv $UV_HOME && \
    $UV_HOME/bin/pip install uv==0.12.*

# Copy from the cache instead of linking (required for using cache mount)
ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=$HOME/.cache/uv

# Install project dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,uid=$UID,target=$UV_CACHE_DIR \
    uv sync --locked --no-dev --extra deploy --no-install-project

# Install the project itself
COPY README.md README.md
COPY app app
COPY conf conf
COPY lulc lulc

RUN --mount=type=cache,uid=$UID,target=$UV_CACHE_DIR \
    uv sync --locked --no-dev --extra deploy

# Deployment stage: a smaller image with only the required files
FROM python_base AS deployment

COPY --from=builder $WD $WD

ENV PATH="$WD/.venv/bin:$PATH"

ENTRYPOINT ["lulc"]

EXPOSE 8000
