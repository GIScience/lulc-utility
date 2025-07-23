FROM ghcr.io/astral-sh/uv:debian

WORKDIR /lulc-utility

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cace/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY app app
COPY conf conf
COPY data data
COPY lulc lulc
COPY data/example cache/sentinelhub/imagery_v1

# Install the project source code separately from its dependencies for optimal layer caching
COPY . /lulc-utility
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group deploy

ENV TRANSFORMERS_CACHE='/tmp'
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
CMD ["uv", "run", "python", "app/api.py"]

EXPOSE 8000
