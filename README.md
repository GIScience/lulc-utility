# LULC utility

This utility within the [Climate Action Framework](https://heigit.atlassian.net/wiki/spaces/CA/pages/170066046/Architecture) provides the full workflow of

 1. training a deep learning model for land-use and land-cover classes,
 2. registering and tracking the model in an online store and
 3. serving the models predictions via a REST-API.

OpenStreetMap LULC polygons are used as training labels via [this OSM2LULC mapping](data/label/label_v1.csv). The model is trained locally and stored at [neptune.ai](https://app.neptune.ai/o/HeiGIT/org/climate-action/models). The feature space is constructed from [Sentinel 1, 2](https://sentinel.esa.int/web/sentinel/missions) and a DEM. The utility can then spawn a [FastAPI](https://fastapi.tiangolo.com/) endpoint which enables operators to request LULC classifications for arbitrary areas and timestamps.

## Install

This Package uses [mamba](https://mamba.readthedocs.io/en/latest/installation.html)  for environment management.
Run `mamba env create -f environment.yaml` to create the environment.

We highly suggest to use a [CUDA](https://en.wikipedia.org/wiki/CUDA)-compatible device to train the model.
To check whether such a device is available on your machine run `nvidia-smi` in the console.

## Data

To select the area on which the model will be trained, an **area descriptor** has to be prepared or computed. The area
descriptor will generate a set of tiles to use during training.
To automatically prepare the descriptor set relevant area parameters
in [`conf/area_descriptor.yaml`](conf/area_descriptor.yaml) and run the following command:

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python lulc/compute_area_descriptor.py
```

## Train

Before training, please check whether you acquired access to the external resources: [Neptune.ai](https://neptune.ai/)
and [SentinelHub](https://www.sentinel-hub.com/).
Following environmental variables have to beset before running the training script:

| ENV                       | Description                                              |
|---------------------------|----------------------------------------------------------|
| NEPTUNE_PROJECT_API_TOKEN | Token acquired from Neptune.ai dashboard (Get Token)     |
| NEPTUNE_PROJECT_ID        | HeiGIT project id acquired from the Neptune.ai dashboard |
| NEPTUNE_MODE              | Any of https://docs.neptune.ai/api/connection_modes/     |
| SENTINELHUB_API_ID        | Id acquired from SentinelHub dashboard                   |
| SENTINELHUB_API_SECRET    | Token acquired from SentinelHub dashboard                |

Training process can be parametrized using relevant configuration files. Visit [`./conf/*.yaml`](conf) for reference.

Training can be run with following commands (project root as working DIR):

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python lulc/train.py
```

## Serve

This will spawn a REST-API that can serve the predictions for the model trained above. 
Before starting the API, please check whether the same environmental variables as in [Train](#train) are set [^1].
To serve the machine learning model choose the desired model version from
the [Model Registry](https://app.neptune.ai/o/HeiGIT/org/climate-action/models?shortId=CA-LULC&type=model),
e.g.: `LULC-SEG-2` and modify the [`conf/serve/local.yaml`](conf/serve/local.yaml) file. Then start the application:

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python app/api.py
```

Go to [localhost:8000/docs](http://localhost:8000/docs) to see the API in action.

[^1]: Note that `debug` mode does not work. We suggest using `read-only` for testing.


---
<img src="docs/logo.png"  width="40%">