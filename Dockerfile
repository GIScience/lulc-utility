FROM ghcr.io/astral-sh/uv:0.8-debian

WORKDIR /lulc-utility

# Enable bytecode compilation for faster application startups
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Temporarily mount uv.lock and pyproject.toml as we don't need them in runtime and they trigger package downloads
# on startup if included in the filesystem
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --group deploy --no-install-project

COPY README.md README.md
COPY app app
COPY conf conf
COPY data data
COPY lulc lulc

# Install the project source code separately from its dependencies for optimal layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --group deploy

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["uv", "run", "python", "app/api.py"]

EXPOSE 8000
