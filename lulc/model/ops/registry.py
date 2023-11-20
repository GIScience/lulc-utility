import logging
from pathlib import Path

import neptune
import torch
from neptune.exceptions import NeptuneModelKeyAlreadyExistsError

from lulc.model.model import SegformerModule

log = logging.getLogger(__name__)


class NeptuneModelRegistry:
    """
    Enables interaction with Neptune.ai model operations
    More: https://docs.neptune.ai/model_registry/overview/
    """

    def __init__(self, model_key: str, project: str, api_key: str, cache_dir: Path):
        self.__model_key = model_key
        self.__project = project
        self.__api_token = api_key
        self.__cache_dir = cache_dir / 'ops'
        self.__cache_dir.mkdir(parents=True, exist_ok=True)

    def register_version(self, model: SegformerModule, run_name: str, run_url: str, label_descriptor_version: str):
        try:
            neptune.init_model(
                name='Climate Action - LULC - SegFormer',
                project=self.__project,
                api_token=self.__api_token,
                key=self.__model_key
            ).sync()
        except NeptuneModelKeyAlreadyExistsError:
            log.info(f'Model {self.__model_key} already exists in Neptune.ai')

        model_path = f'{self.__cache_dir}/{run_name}.onnx'
        log.info(f'Persisting temporary onnx model file in: {model_path}')

        model.to_onnx(model_path,
                      input_sample=torch.zeros((1, model.configuration.num_channels, 1024, 1024)),
                      dynamic_axes={'imagery': [2, 3], 'labels': [1, 2]},
                      input_names=['imagery'],
                      output_names=['labels'])

        project_id = neptune.init_project(project=self.__project, api_token=self.__api_token, mode='read-only')['sys/id'].fetch()
        neptune_model = f'{project_id}-{self.__model_key}'
        log.info(f'Registering model version: {neptune_model}')
        model_version = neptune.init_model_version(
            name=run_name,
            model=neptune_model,
            project=self.__project,
            api_token=self.__api_token,
        )
        model_version['run/url'] = run_url
        model_version['model'].upload(model_path)
        model_version['label_descriptor_version'].upload(label_descriptor_version)
        model_version.sync()

        log.info(f'Model {neptune_model} version has been registered: {model_version.get_url()}')
