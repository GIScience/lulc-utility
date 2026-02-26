import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import torch

from lulc.model.model import SegformerModule

log = logging.getLogger(__name__)


class MLflowModelRegistry:
    """
    Enables interaction with a MLflow experiment tracking and model registry backend. Requires the
    `MLFLOW_TRACKING_TOKEN` environment variable to be set, and the model to be registered in the model registry.
    """

    def __init__(self, tracking_uri: str, model_name: str, cache_dir: Path):
        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_client = mlflow.MlflowClient()
        self.model_name = model_name
        self.cache_dir = cache_dir / 'ops' / model_name
        if not self.cache_dir.exists():
            os.makedirs(self.cache_dir, exist_ok=True)

    def register_version(self, model: SegformerModule, run_name: str, run_id: str) -> None:
        model_path = self.cache_dir / f'{run_name}.onnx'
        log.info(f'Persisting temporary onnx model file in: {model_path}')
        onnx_model = model.to_onnx(
            input_sample=torch.zeros((1, model.configuration.num_channels, 1024, 1024)),
            dynamic_axes={'imagery': [2, 3], 'labels': [1, 2]},
            input_names=['imagery'],
            output_names=['labels'],
        )

        with mlflow.start_run(run_id=run_id):
            mlflow.onnx.log_model(
                onnx_model=onnx_model.model_proto,
                artifact_path='',
                input_example=np.zeros((1, model.configuration.num_channels, 1024, 1024)),
            )

        log.info(f'Model has been uploaded to run {run_id}. Visit the UI to promote the run to a model.')

    def download_model(self, model_version: str = 'latest') -> Path:
        """Download the requested model_version from the GitLab model registry and return the path to the downloaded
        ONNX model file.
        """
        model_path = self.cache_dir / model_version
        os.makedirs(model_path, exist_ok=True)

        onnx_model = mlflow.onnx.load_model(f'models:/{self.model_name}/{model_version}', dst_path=model_path)
        return onnx_model.SerializeToString()
