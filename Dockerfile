ARG BASE_IMAGE=python:3.13-slim-bookworm

# Build stage: having a build stage means temp/cache files from the build aren't persisted in the final image
FROM $BASE_IMAGE AS builder

ARG HOME=/root

# Install uv in an isolated venv to avoid conflicts with the package venv
ARG UV_HOME=$HOME/uv
RUN --mount=type=cache,target=$HOME/.cache/pip \
    python3 -m venv $UV_HOME && \
    $UV_HOME/bin/pip install uv==0.12.*
ENV PATH="$UV_HOME/bin:$PATH"

# Copy from the cache instead of linking (required for using cache mount)
ARG UV_LINK_MODE=copy
ARG UV_CACHE_DIR=$HOME/.cache/uv

# Require git to install git dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends git

WORKDIR /home/lulc

# Install project dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --locked --no-dev --extra deploy --no-install-project

# Copy the project files
COPY README.md README.md
COPY app app
COPY conf conf
COPY lulc lulc

# Install the project itself
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --locked --no-dev --extra deploy

# Deployment stage: a smaller image with only the required files
FROM $BASE_IMAGE AS deployment

# Install system-level shared libs needed by compiled deps (e.g. rasterio -> libexpat)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libexpat1

# Create a dedicated user to run all tasks (as non-root)
ENV USER=lulc
ARG UID=99
ARG GID=$UID
RUN addgroup lulc --gid $GID && \
    useradd -u $UID -g $GID -ms /bin/bash $USER
USER $USER

# Must be same as in builder image, see above
ENV WD=/home/lulc
WORKDIR $WD

COPY --from=builder --chown=$USER $WD $WD
ENV PATH="$WD/.venv/bin:$PATH"

ENTRYPOINT ["lulc"]
EXPOSE 8000
