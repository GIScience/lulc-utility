import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from sentinelhub import BBox, CRS, bbox_to_dimensions, SentinelHubRequest, DataCollection, MosaickingOrder, MimeType, \
    SHConfig, DownloadFailedException

from lulc.ops.exception import OperatorInteractionException, OperatorValidationException

log = logging.getLogger(__name__)


class ImageryStore(ABC):

    @abstractmethod
    def imagery(self, area_coords: Tuple, start_date: str, end_date: str,
                resolution: int = 10) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        pass


class SentinelHubOperator(ImageryStore):

    def __init__(self, api_id: str, api_secret: str, evalscript_dir: Path, evalscript_name: str, cache_dir: Path):
        self.config = SHConfig(**{
            'sh_client_id': api_id,
            'sh_client_secret': api_secret
        })
        self.cache_dir: Path = cache_dir
        self.evalscript = (evalscript_dir / f'{evalscript_name}.js').read_text()

        self.data_folder = self.cache_dir / evalscript_name
        self.data_folder.mkdir(parents=True, exist_ok=True)

    def imagery(self, area_coords: Tuple, start_date: str, end_date: str,
                resolution: int = 10) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
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
                    time_interval=(start_date, end_date)
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    identifier='s2',
                    time_interval=(start_date, end_date),
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.DEM,
                    identifier='dem',
                    time_interval=(start_date, end_date),
                )
            ],
            responses=[
                SentinelHubRequest.output_response('s1', MimeType.TIFF),
                SentinelHubRequest.output_response('s2', MimeType.TIFF),
                SentinelHubRequest.output_response('dem', MimeType.TIFF)
            ],
            bbox=bbox,
            size=(bbox_width, bbox_height),
            config=self.config,
        )

        try:
            return request.get_data(save_data=True)[0], (bbox_height, bbox_width)
        except DownloadFailedException:
            log.exception('Data download not possible')
            raise OperatorInteractionException('SentinelHub operator interaction not possible. Please contact platform administrator.')
