import logging
from pathlib import Path
from typing import Optional

import neptune

log = logging.getLogger(__name__)


class NeptuneModelDownload:
    """
    Class for automated download of machine learning models stored at Neptune.ai.
    More: https://docs.neptune.ai/model_registry/overview/
    """

    def __init__(self, model_key: str, project: str, api_key: str, cache_dir: Path):
        self.__model_key = model_key
        self.__project = project
        self.__api_token = api_key
        self.__cache_dir = cache_dir / 'ops'
        self.__cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model_version(self, neptune_model_version_id: Optional[str]) -> Path:
        if neptune_model_version_id is None:
            neptune_model_version_id = self.__fetch_latest_by_state()

        log.info(f'Utilizing model {neptune_model_version_id}')

        model_file = self.__cache_dir / f'{neptune_model_version_id}.onnx'
        if not model_file.exists():
            model_version = neptune.init_model_version(
                with_id=neptune_model_version_id,
                project=self.__project,
                api_token=self.__api_token,
            )
            model_version['model'].download(str(model_file))
        return model_file

    def __fetch_latest_by_state(self, state='production'):
        project_id = neptune.init_project(project=self.__project, api_token=self.__api_token, mode='read-only')['sys/id'].fetch()
        model = neptune.init_model(with_id=f'{project_id}-{self.__model_key}', project=self.__project, api_token=self.__api_token)
        model_versions_df = model.fetch_model_versions_table().to_pandas()
        production_models = model_versions_df[model_versions_df['sys/stage'] == state]

        if production_models.empty:
            raise ValueError(f'No {state} model available in project {self.__project}')

        return production_models.sort_values(by=['sys/creation_time'], ascending=False).iloc[0]['sys/id']
