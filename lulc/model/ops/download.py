import logging
from pathlib import Path

import neptune

log = logging.getLogger(__name__)


class NeptuneModelDownload:
    """
    Enables interaction with Neptune.ai model ops
    More: https://docs.neptune.ai/model_registry/overview/
    """

    def __init__(self, model_key: str, project: str, api_key: str, cache_dir: Path):
        self.__model_key = model_key
        self.__project = project
        self.__api_token = api_key
        self.__cache_dir = cache_dir / 'ops'
        self.__cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model_version(self, neptune_model_version_id: str) -> Path:
        model_file = self.__cache_dir / f'{neptune_model_version_id}.onnx'
        if not model_file.exists():
            model_version = neptune.init_model_version(
                with_id=neptune_model_version_id,
                project=self.__project,
                api_token=self.__api_token,
            )
            model_version['model'].download(str(model_file))
        return model_file
