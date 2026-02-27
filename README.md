# LULC utility

This utility within
the [Climate Action Framework](https://heigit.atlassian.net/wiki/spaces/CA/pages/170066046/Architecture) provides the
full workflow of:

1. training a deep learning model for land-use and land-cover classes,
2. registering and tracking the model in an online store and
3. serving the model's predictions via a REST API which enables users to request LULC
   classifications for arbitrary areas and timestamps.

The main reason for creating such a utility is to:

1. Gain access to a mechanism capable of preparing semantic segmentation models fine-tailored for specific research
   scenarios,
2. Enable rapid model training for various sets of labels (encoded as OSM query filters) and circumstances (
   unpredictable events),
3. Possibility to use the model as a high-quality imputing mechanism (missing data filling) for real-time OSM data.
4. Create a lightweight solution that can be adjusted for various types of hardware,
5. Prepare a project baseline that can be configured to utilize different data sources and model implementation.

## Install

This Package uses [uv](https://docs.astral.sh/uv/c) for environment and package management.
To install dependencies run `uv sync`.
By default, only the dependencies common to model training and serving are installed.
To install dependencies required for either of them run

```shell
uv sync --extra deploy  # to only install deployment dependencies
uv sync --extra train  # to only install model training dependencies
uv sync --all-extras  # to install all extra dependencies
```


We highly suggest using a [CUDA](https://en.wikipedia.org/wiki/CUDA)-a compatible device to train the model.
To check whether such a device is available on your machine, run `nvidia-smi` in the console.
When you start the training script, the logs will include a line like `GPU available: True (cuda), used: True` if CUDA
is available.

## Configuration

To use the utility, you require access to several external services, depending on the model run.
Review the secret keys and common configuration variables in the [`.env.template`](.env_template) and create a local
`.env` file accordingly.
Any environment variables that include special characters must be wrapped in single quotation marks (`''`).

There are many more configuration files available in [`/conf/`](/conf/).
Visit [`/conf/config.yaml`](/conf/config.yaml) to review and modify the configuration as required.
Note that the config under `train` relates to training-specific configuration, while the config under `serve` is used
by the inference API and should only be modified with caution.

### Model Registry

The utility is currently configured to use the model registry and experiment tracking
of [GitLab](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/ml/models).
To work with this backend, you need to configure the tracking URI and tracking token.

The **tracking URI** is set in the config files, and is specific to this repository in GitLab.
As long as you are working in **this** repository, you can use the default tracking URI.
If you clone this to another location, or want to store your models in another repository, you should get the
appropriate tracking URI on GitLab by going to _Deploy > Model Registry > Create/Import Model > Import model using
MLflow_ and copying the displayed `MLFLOW_TRACKING_URI`.

The **tracking token** must be set as an environment variable, `MLFLOW_TRACKING_TOKEN`.
To create a tracking token, in GitLab go to _Your avatar (top right) > Preferences > Personal access tokens > Add new
token_ and create a token with `api` access.

## Development

Note that the repository supports pre commit hooks defined in the `.pre-commit-config.yaml` file.
For the description of each hook visit the documentation of:

- [git pre-commit hooks](https://github.com/pre-commit/pre-commit-hooks)
- [ruff pre-commit hooks](https://github.com/astral-sh/ruff-pre-commit)

Run `uv run pre-commit install` to activate them.

### Testing

Run `uv run pytest` to run the tests.
Note that the tests require **all** optional dependency groups.

## Training

**The following pipeline is for training models and requires the optional `train` dependencies.**

OpenStreetMap LULC polygons are used as training labels via [this OSM2LULC mapping](data/label/label_v3.yaml).
The model is trained locally and stored
on [GitLab](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/ml/models).
The feature space is constructed from [Sentinel 1, 2](https://sentinel.esa.int/web/sentinel/missions) and a DEM.

The underlying semantic segmentation model is called [SegFormer](https://arxiv.org/abs/2105.15203).
It has the ability to delineate homogenous regions precisely.
Due to its configurable and lightweight architecture, SegFormer can be quickly trained to match a specific use case.
Preparing a country-specific model should take around two days (GPU: GeForce 3090).

### Data preparation

To start training a new model, first create a config folder under [`/conf/train/`](/conf/train/) and update the default
config path in [`/conf/config.yaml`](/conf/config.yaml).
Then copy the relevant config files from [`/conf/examples/*.yaml`](/conf/examples/) and follow the comments from the
examples to create your config files.

After initial configuration of your training parameters, the following data preparation steps have been automated in
scripts, as described below.

#### Area description

To select the area on which the model will be trained, an **area descriptor** has to be prepared or computed.
The area descriptor will generate a set of tiles to use during training.
To automatically prepare the descriptor, set relevant area parameters
in `conf/train/<YOUR_FOLDER>/area_descriptor.yaml` and run the following command:

```shell
uv run lulc/compute_area_descriptor.py
```

The area descriptor (as a `csv`) and a visual representation of it (as `png`) will be saved
to [`data/area/`](data/area/).

Optionally, to sanity check the imagery that would be sourced for each of the tiles, run:

```shell
uv run --env-file .env lulc/save_imagery.py
```

To save just a small sample area, you can also create a smaller area descriptor file and provide the optional command
line input `'train.area.aoi_file=data/area/test.csv'` (note the string wrapping).

Note that the `sentinel_hub` operator already caches the imagery requested, so if using Sentinel Hub, one can also find
the relevant tiff files in `/cache/imagery/sentinel_hub/...`.

#### Ground truth labels

OpenStreetMap LULC polygons are used as ground truth training labels via the OSM2LULC mapping defined in
`/conf/train/<YOUR_FOLDER>/label.yaml`.
To define the ground truth labels, create a new `label.yaml` file in your training config.

Optionally, to export a single raster of your ground truth labels (for visual inspection and approval) of your whole
training area, run:

```shell
uv run --env-file .env lulc/export_osm_labels.py
```

The raster files of ground truth labels will be saved to the cache_dir.

#### Normalization and class weights

Images need to be normalised to a similar range across sensor channels before they are used during model training.

LULC data also always contains class imbalance.
In OSM, this imbalance can be aggravated through the data collection process.
The model can make use of class weights to account for this problem by adjusting the loss function.
The class weights are based on the spatial resolution of the training data and the resulting number of pixels that are
attributed to each class.

To calculate a reasonable set of normalisation values and class weights for new datasets, one needs to run the following
script (note that this will load all of the images in your area descriptor so can take some time):

```shell
uv run --env-file .env lulc/calculate_dataset_statistics.py
```

The resulting image normalisation parameters and class weights must be copied to `data.yaml` before training.

### Run

Finally, after setting up the configuration, training can be run with the following command:

```shell
uv run --env-file .env lulc/train.py
```

## Serve

**The following pipeline is for deploying the API and requires the optional `deploy` dependencies.**

To serve an inference session for a model trained in this utility, we spawn a REST API locally.
Before starting the API, the configuration in [`/conf/serve/`](/conf/serve/) must be updated for the model being hosted.
The relevant config files can simply be copied from the matching training configuration, and then changing the
`# @package train.XYZ` header to `# @package serve.XYZ`.

Also choose the desired model version from
the [Model Registry](https://gitlab.heigit.org/climate-action/utilities/lulc-utility/-/ml/models),
e.g.: `1.0.0` and set the `MODEL_VERSION` in your `.env` file accordingly.

Then start the application:

```shell
uv run --env-file .env app/api.py
```

> Go to [localhost:8000](http://localhost:8000/docs) to see the API in action.

### Docker

The tool is also [Dockerised](Dockerfile).
Images are automatically built and deployed in the [CI-pipeline](.gitlab-ci.yml).

In case you want to manually build and run locally (e.g. to test a new feature in development), execute

```shell
docker build . --tag repo.heigit.org/climate-action/lulc-utility:devel
docker run --rm --publish 8000:8000 --env-file .env repo.heigit.org/climate-action/lulc-utility:devel
```

Note that this will overwrite any existing image with the same tag (i.e. the one you previously pulled from the Climate
Action docker registry).

To run behind a proxy, you can configure the root path using the environment variable `ROOT_PATH`.

#### Deploy

To push a new version to [our docker registry](repo.heigit.org) (i.e. to overwrite the one on the Climate
Action docker registry), run

```shell
docker build . --tag repo.heigit.org/climate-action/lulc-utility:devel
docker image push repo.heigit.org/climate-action/lulc-utility:devel
```

## Releasing a new utility version

1. Update the [CHANGELOG.md](CHANGELOG.md).
   It should already be up to date but give it one last read and update the heading above this upcoming release
2. Decide on the new version number.
   Please adhere to the [Semantic Versioning](https://semver.org/) scheme, based on the changes since the last release.
3. Update the version attribute in the [pyproject.toml](pyproject.toml)
4. Create a [release]((https://docs.gitlab.com/ee/user/project/releases/#create-a-release-in-the-releases-page)) on
   GitLab, including a changelog

---
<img src="docs/logo.png"  width="40%">
