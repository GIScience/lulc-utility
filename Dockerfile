FROM condaforge/mambaforge:23.1.0-4 AS build

COPY environment_deploy.yaml environment.yaml

RUN mamba env create -f environment.yaml && \
    mamba install -c conda-forge conda-pack && \
    conda-pack -f --ignore-missing-files -n ca-lulc-utility -o /tmp/env.tar && \
    mkdir /venv && \
    cd /venv && \
    tar xf /tmp/env.tar && \
    rm /tmp/env.tar  && \
    /venv/bin/conda-unpack && \
    mamba clean --all --yes

FROM debian:buster AS runtime

WORKDIR /ca-lulc-utility
COPY --from=build /venv /ca-lulc-utility/venv

COPY app app
COPY conf conf
COPY data data
COPY lulc lulc
COPY data/example cache/sentinelhub/imagery_v1

ENV TRANSFORMERS_CACHE='/tmp'
ENV PYTHONPATH "${PYTHONPATH}:/ca-lulc-utility/lulc"

SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /ca-lulc-utility/venv/bin/activate && \
           uvicorn app.api:app --host 0.0.0.0 --port 8000 --root-path '/api/lulc/v1' --log-config=conf/logging/app/logging.yaml
