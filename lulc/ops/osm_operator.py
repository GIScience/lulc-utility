import hashlib
import uuid
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import rasterio
from geocube.api.core import make_geocube
from ohsome import OhsomeClient
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.enums import Resampling


class OhsomeOps:

    def __init__(self, cache_dir: Path, resolution=(-.0001, .0001)):
        self.cache_dir: Path = cache_dir
        self.resolution = resolution
        self.ohsome = OhsomeClient(user_agent='ClimateAction/LULC')

    def labels(self, area_coords: Tuple[float, float, float, float],
               time: str,
               osm_lulc_mapping: Dict,
               target_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        result = {}
        height, width = target_size

        for label, osm_filter in osm_lulc_mapping.items():
            bbox = ','.join(map(str, area_coords))
            bbox_id = self.__calculate_id(bbox)

            data_folder = self.cache_dir / label
            data_folder.mkdir(parents=True, exist_ok=True)
            raster_data = data_folder / f'{bbox_id}.tiff'

            if not raster_data.exists():
                response = self.ohsome.elements.geometry.post(
                    bboxes=bbox,
                    time=time,
                    filter=f'({osm_filter}) and geometry:polygon'
                )

                vector_data = response.as_dataframe()
                if not vector_data.empty:
                    utm = self.__utm_from_coords(area_coords)
                    raster = self.__computer_raster(vector_data, utm)
                    raster.rio.to_raster(raster_data)

            if raster_data.exists():
                with rasterio.open(raster_data) as dataset:
                    data = dataset.read(1, out_shape=(dataset.count, height, width), resampling=Resampling.bilinear)
            else:
                data = np.zeros((dataset.count, height, width))

            result[label] = data.astype(np.bool_)

        return result

    @staticmethod
    def __calculate_id(text: str) -> uuid.UUID:
        hex_string = hashlib.md5(text.encode("UTF-8")).hexdigest()
        return uuid.UUID(hex=hex_string)

    def __computer_raster(self, vector_data, utm):
        vector_data['value'] = 1
        sorted_desc_areas_idx = vector_data.copy().to_crs(utm).geometry.area.argsort()[::-1]

        return make_geocube(
            vector_data=vector_data.iloc[sorted_desc_areas_idx],
            measurements=['value'],
            resolution=self.resolution,
            output_crs='EPSG:4326',
            fill=0
        ).astype(np.uint8)

    @staticmethod
    def __utm_from_coords(area_coords: Tuple[float, float, float, float]) -> str:
        utm = query_utm_crs_info(datum_name='WGS 84', area_of_interest=AreaOfInterest(*area_coords))
        return f'{utm[0].auth_name}:{utm[0].code}'
