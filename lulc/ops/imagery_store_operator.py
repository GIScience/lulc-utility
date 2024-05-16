import logging
import logging as rio_logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from tarfile import ReadError
from typing import Dict, Tuple

import numpy as np
from omegaconf import DictConfig
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadFailedException,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

from lulc.ops.exception import OperatorInteractionException, OperatorValidationException

log = logging.getLogger(__name__)
rio_logging.getLogger('rasterio._filepath').setLevel(logging.ERROR)


class ImageryStore(ABC):
    @abstractmethod
    def imagery(
        self, area_coords: Tuple, start_date: str, end_date: str, resolution: int = 10
    ) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        pass

    def labels(
        self,
        area_coords: Tuple,
        cor_api_id: str,
        catalog_id: str,
        service_url: str,
        evalscript_dir: Path,
        evalscript_name: str,
        start_date: str,
        end_date: str,
        evalscript_name_cor: str,
        corine_years: list,
        cache_dir: Path,
        resolution: int = 100,
    ) -> (tuple)[Dict[str, np.ndarray], tuple[int, int]]:
        pass


class SentinelHubOperator(ImageryStore):
    def __init__(
        self,
        api_id: str,
        api_secret: str,
        evalscript_dir: Path,
        evalscript_name: str,
        cor_api_id: str,
        catalog_id: str,
        service_url: str,
        evalscript_name_cor: str,
        corine_years: list,
        cache_dir: Path,
    ):
        self.config = SHConfig(**{'sh_client_id': api_id, 'sh_client_secret': api_secret})
        self.cache_dir: Path = cache_dir / 'sentinel_hub'
        self.evalscript_dir = evalscript_dir
        self.evalscript = (evalscript_dir / f'{evalscript_name}.js').read_text()

        self.data_folder = self.cache_dir / evalscript_name
        self.data_folder.mkdir(parents=True, exist_ok=True)

        self.cor_api_id = cor_api_id
        self.catalog_id = catalog_id
        self.service_url = service_url
        self.evalscript_cor = (evalscript_dir / f'{evalscript_name_cor}.js').read_text()

        self.data_folder_cor = self.cache_dir / evalscript_name_cor
        self.data_folder_cor.mkdir(parents=True, exist_ok=True)

        self.corine_years = corine_years

    def imagery(
        self,
        area_coords: Tuple,
        start_date: str,
        end_date: str,
        resolution: int = 10,
    ) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        bbox = BBox(bbox=area_coords, crs=CRS.WGS84)
        bbox_width, bbox_height = bbox_to_dimensions(bbox, resolution=resolution)

        if bbox_width > 2500 or bbox_width > 2500:
            raise OperatorValidationException('Area exceeds processing limit: 2500 px x 2500 px')

        request = SentinelHubRequest(
            data_folder=str(self.data_folder),
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1,
                    identifier='s1',
                    time_interval=(start_date, end_date),
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    identifier='s2',
                    time_interval=(start_date, end_date),
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.DEM,
                    identifier='dem',
                    time_interval=(start_date, end_date),
                ),
            ],
            responses=[
                SentinelHubRequest.output_response('s1', MimeType.TIFF),
                SentinelHubRequest.output_response('s2', MimeType.TIFF),
                SentinelHubRequest.output_response('dem', MimeType.TIFF),
            ],
            bbox=bbox,
            size=(bbox_width, bbox_height),
            config=self.config,
        )

        try:
            return request.get_data(save_data=True)[0], (bbox_height, bbox_width)
        except DownloadFailedException:
            log.exception('Data download not possible')
            raise OperatorInteractionException(
                'SentinelHub operator interaction not possible. Please contact platform administrator.'
            )
        except ReadError:
            invalid_item = Path(request.get_filename_list()[0]).parent
            log.warning(f'Cached TAR file read error (cache id: {invalid_item}) - dropping and retrying download')

            shutil.rmtree(Path(request.data_folder) / invalid_item)
            return request.get_data(save_data=True)[0], (bbox_height, bbox_width)

    def labels(
        self,
        area_coords: Tuple,
        start_date: str,
        end_date: str,
        resolution: int = 100,
    ) -> (tuple)[Dict[str, np.ndarray], tuple[int, int]]:
        bbox = BBox(bbox=area_coords, crs=CRS.WGS84)
        bbox_width, bbox_height = bbox_to_dimensions(bbox, resolution=resolution)

        corine_years = self.corine_years

        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

        filtered_years = list(filter(lambda year: year <= end_date_dt.year, corine_years))

        corine_end_year = filtered_years[-1]
        corine_date_str = f'{corine_end_year}-01-01'
        start_date = corine_date_str
        end_date = corine_date_str

        if bbox_width > 2500 or bbox_height > 2500:
            raise OperatorValidationException('Area exceeds processing limit: 2500 px x 2500 px')

        request = SentinelHubRequest(
            data_folder=str(self.data_folder_cor),
            evalscript=self.evalscript_cor,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.define(
                        name='clc',
                        api_id=self.cor_api_id,
                        catalog_id=self.catalog_id,
                        service_url=self.service_url,
                        is_timeless=False,
                    ),
                    time_interval=(start_date, end_date),
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF),
            ],
            bbox=bbox,
            size=(bbox_width, bbox_height),
            config=self.config,
        )

        try:
            return request.get_data(save_data=True)[0], (bbox_height, bbox_width)
        except DownloadFailedException:
            log.exception('Data download not possible')
            raise OperatorInteractionException(
                'Corine operator interaction not possible. Please contact platform administrator.'
            )


def resolve_imagery_store(cfg: DictConfig, cache_dir: Path) -> ImageryStore:
    if cfg.operator == 'sentinel_hub':
        return SentinelHubOperator(
            cfg.api_id,
            cfg.api_secret,
            Path(cfg.evalscript_dir),
            cfg.evalscript_name,
            cfg.cor_api_id,
            cfg.catalog_id,
            cfg.service_url,
            cfg.evalscript_name_cor,
            cfg.corine_years,
            cache_dir=cache_dir / 'imagery',
        )
    else:
        raise ValueError(f'Cannot resolve imagery operator {cfg.operator}')
