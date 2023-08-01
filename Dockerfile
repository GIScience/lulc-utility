FROM mambaorg/micromamba:1.4.9-focal-cuda-11.7.1

USER root
RUN apt update && \
    apt install -y libsm6 libxext6

USER $MAMBA_USER

COPY environment.yaml .

RUN micromamba install -y -n base -f environment.yaml -v && \
    micromamba clean --all --yes

USER root
WORKDIR /ca-lulc-utility

COPY app app
COPY conf conf
COPY data data
COPY lulc lulc
COPY example/data cache/sentinelhub/imagery_v1

RUN useradd lulc && \
    chown -R lulc /ca-lulc-utility

USER lulc

ENV TRANSFORMERS_CACHE='/tmp'
ENV PYTHONPATH "${PYTHONPATH}:/ca-lulc-utility/lulc"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
