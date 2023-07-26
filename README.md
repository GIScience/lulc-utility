# LULC utility

## Install

This Package uses [mamba](https://mamba.readthedocs.io/en/latest/installation.html)  for environment management.
Run `mamba env create -f environment.yaml` to create the environment.

Please note that you will need to use a [CUDA](https://en.wikipedia.org/wiki/CUDA)-compatible device to train the model.
To check whether such a device is available on your machine run `nvidia-smi` in the console.

## Data

To select the area on which the model will be trained, an **area descriptor** has to be prepared or computed. The area
descriptor will generate a set of tiles to use during training.
To automatically prepare the descriptor set relevant area parameters in [`conf/area_descriptor.yaml`](conf/area_descriptor.yaml) and run the following command:

```bash
export PYTHONPATH="lulc:$PYTHONPATH"
python lulc/compute_area_descriptor.py
```

## Train

Before training please check whether you acquire access to external resources: [Neptune.ai](https://neptune.ai/)
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

## Deploy

---
<img src="docs/logo.png"  width="40%">