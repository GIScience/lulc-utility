FROM condaforge/mambaforge:24.9.2-0 AS build

COPY environment_deploy.yaml environment.yaml

RUN mamba env create -f environment.yaml && \
    mamba install -y -c conda-forge conda-pack && \
    conda-pack -f --ignore-missing-files -n ca-lulc-utility-deploy -o /tmp/env.tar && \
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
           python app/api.py
EXPOSE 8000
