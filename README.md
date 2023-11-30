# LULC utility

This utility within
the [Climate Action Framework](https://heigit.atlassian.net/wiki/spaces/CA/pages/170066046/Architecture) provides the
full workflow of:

1. training a deep learning model for land-use and land-cover classes,
2. registering and tracking the model in an online store and
3. serving the model's predictions via a REST API.

The main reason for creating such a utility is to:

1. Gain access to a mechanism capable of preparing semantic segmentation models fine-tailored for specific research
   scenarios,
2. Enable rapid model training for various sets of labels (encoded as OSM query filters) and circumstances (
   unpredictable events),
3. Possibility to use the model as a high-quality imputing mechanism (missing data filling) for real-time OSM data.
4. Create a lightweight solution that can be adjusted for various types of hardware,
5. Prepare a project baseline that can be configured to utilize different data sources and model implementation.

## Methods

OpenStreetMap LULC polygons are used as training labels via [this OSM2LULC mapping](data/label/label_v3.yaml).
The model is trained locally and stored at [neptune.ai](https://app.neptune.ai/o/HeiGIT/org/climate-action/models). The
feature space is constructed from [Sentinel 1, 2](https://sentinel.esa.int/web/sentinel/missions) and a DEM.
The utility can then spawn a [FastAPI](https://fastapi.tiangolo.com/) endpoint which enables operators to request LULC
classifications for arbitrary areas and timestamps.

The underlying semantic segmentation model is called [SegFormer](https://arxiv.org/abs/2105.15203).
It has the ability to delineate homogenous regions precisely.
Due to its configurable and lightweight architecture, SegFormer can be quickly trained to match a specific use case.
Preparing a country-specific model should take around two days (GPU: GeForce 3090).

## Install

This Package uses [mamba](https://mamba.readthedocs.io/en/latest/installation.html)  for environment management.
Run `mamba env create -f environment.yaml` to create the environment.

We highly suggest using a [CUDA](https://en.wikipedia.org/wiki/CUDA)-a compatible device to train the model.
To check whether such a device is available on your machine, run `nvidia-smi` in the console.

## Development

Note that the repository supports pre commit hooks defined in the `.pre-commit-config.yaml` file.
For the description of each hook visit the documentation of:

- [git pre-commit hooks](https://github.com/pre-commit/pre-commit-hooks)
- [ruff pre-commit hooks](https://github.com/astral-sh/ruff-pre-commit)

Run `pre commit install` to activate them.

## Data

### Area

To select the area on which the model will be trained, an **area descriptor** has to be prepared or computed.
The area descriptor will generate a set of tiles to use during training.
To automatically prepare the descriptor set relevant area parameters
in [`conf/area_descriptor.yaml`](conf/area_descriptor.yaml) and run the following command:

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python lulc/compute_area_descriptor.py
```

### Normalization

Images need to be normalised to a similar range across sensor channels before they are used during model training.
While normalisation for surface reflectance from S2 is straight forward, S1 needs an informed logic for normalisation
, and the DEM should be locally normalised.
Normalisation parameters can be set in the [`conf/data/*.yaml`](conf/data) (`data.normalize`).

To recalculate a reasonable set of values for new datasets, one needs to run the following script:

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python lulc/calculate_dataset_statistics.py
```

### Class weights

LULC data always contains class imbalance.
In OSM, this imbalance can be aggravated through the data collection process.
The model can make use of class weights to account for this problem by adjusting the loss function.
Class weights are declared in the [`conf/data/*.yaml`](conf/data) (`data.class_weights`).

The [script above](#normalization) will also print the suggested class weights.

## Train

### Configuration

Before training, please check whether you acquired access to the external resources: [Neptune.ai](https://neptune.ai/)
and [SentinelHub](https://www.sentinel-hub.com/).
The following environmental variables have to beset before running the training script:

| ENV                       | Description                                              |
|---------------------------|----------------------------------------------------------|
| NEPTUNE_PROJECT_API_TOKEN | Token acquired from Neptune.ai dashboard (Get Token)     |
| NEPTUNE_PROJECT_ID        | HeiGIT project id acquired from the Neptune.ai dashboard |
| NEPTUNE_MODE              | Any of https://docs.neptune.ai/api/connection_modes/     |
| SENTINELHUB_API_ID        | Id acquired from SentinelHub dashboard                   |
| SENTINELHUB_API_SECRET    | Token acquired from SentinelHub dashboard                |
| LOG_LEVEL                 | The minimum level for log messages                       |

The training process can be parametrized using relevant configuration files. Visit [`./conf/**/*.yaml`](conf) for
reference.

### Run

Training can be run with the following commands (project root as working DIR):

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python lulc/train.py
```

## Serve

It will spawn a REST API locally to serve the predictions for the model trained above.
Before starting the API, please check whether the same environmental variables as in [Train](#train) are set [^1].
To serve the machine learning model choose the desired model version from
the [Model Registry](https://app.neptune.ai/o/HeiGIT/org/climate-action/models?shortId=CA-LULC&type=model),
e.g.: `LULC-SEG-2` and modify the [`conf/serve/local.yaml`](conf/serve/local.yaml) file. Then start the application:

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python app/api.py
```

> Go to [localhost:8000](http://localhost:8000) to see the API in action.

### Docker

The tool is also [Dockerised](Dockerfile). To start it, run the following commands

```shell
docker build . --tag heigit/ca-lulc-utility:devel
docker run --publish 8000:8000  --env-file .env heigit/ca-lulc-utility:devel
```

Then head to the link above. Populate the .env file using the [.env_template](.env_template).

To run behind a proxy, you can configure the root path using the environment variable `ROOT_PATH`.

#### Deploy

To push a new version to [Docker Hub](https://hub.docker.com/orgs/heigit) run

```shell
docker build . --tag heigit/ca-lulc-utility:devel
docker image push heigit/ca-lulc-utility:devel
```

[^1]: Note that for Neptune.ai, `debug` mode does not work. We suggest using `read-only` for testing.


---
<img src="docs/logo.png"  width="40%">
