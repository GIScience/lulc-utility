import hashlib
import uuid
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import xarray
from geocube.api.core import make_geocube
from hydra.core.hydra_config import HydraConfig
from ohsome import OhsomeClient
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.enums import Resampling

from lulc.data.label import LabelDescriptor


class OhsomeOps:
    def __init__(self, cache_dir: Path, resolution=(-0.0001, 0.0001)):
        self.cache_dir: Path = cache_dir
        self.resolution = resolution

        log_dir = HydraConfig.get().runtime.output_dir if HydraConfig.initialized() else None
        self.ohsome = OhsomeClient(user_agent='ClimateAction/LULC', log_dir=log_dir, log=HydraConfig.initialized())

    def labels(
        self,
        area_coords: Tuple[float, float, float, float],
        time: str,
        osm_lulc_mapping: Dict[str, LabelDescriptor],
        target_size: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """
        Query the ground truth class labels from OSM features and return a dictionary containing the rasterised labels
        for each class.

        :param area_coords: bounding box coordinates to generate labels for
        :param time: the datetime to query OSM data for (only valid OSM features at this time will be included)
        :param osm_lulc_mapping: a dictionary with keys as class labels and values as ohsome filters for OSM features
        :param target_size: the target dimensions of the rasterised labels
        """
        result = {}
        height, width = target_size
        bbox = ','.join(map(str, area_coords))
        bbox_id = self.__calculate_id(bbox)
        utm = self.__utm_from_coords(area_coords)

        compute_label_mask_p = partial(self.__compute_label_mask, area_coords, bbox_id, time, utm, height, width)
        osm_lulc_mapping = dict([(k, v) for k, v in osm_lulc_mapping.items() if v.osm_filter is not None])

        for item in osm_lulc_mapping.items():
            label, data = compute_label_mask_p(item)
            result[label] = data.astype(np.bool_)

        return result

    def __compute_label_mask(
        self,
        bbox: Tuple[float, float, float, float],
        bbox_id: uuid.UUID,
        time: str,
        utm: str,
        height: int,
        width: int,
        osm_lulc_mapping: Tuple[str, LabelDescriptor],
    ) -> Tuple[str, np.ndarray]:
        """
        Query OSM for the features within the bbox that satisfy the filters defined in the osm_lulc_mapping.

        :param bbox: bounding box coordinates to get labels for
        :param bbox_id: a unique id to save the results to
        :param time: the datetime to query OSM data for (only valid OSM features at this time will be included)
        :param utm: the utm code for the provided bbox
        :param height: the target height for the rasterised labels
        :param width: the target width for the rasterised labels
        :param osm_lulc_mapping: the class name and the ohsome filter to be queried for that class
        """
        label_name, label = osm_lulc_mapping
        data_folder = self.cache_dir / label_name
        data_folder.mkdir(parents=True, exist_ok=True)
        raster_data = data_folder / f'{bbox_id}.tiff'

        if not raster_data.exists():
            filter_geometry = ' or '.join([f'geometry:{geom}' for geom in label.geometry_types])
            vector_data = self.ohsome.elements.geometry.post(
                bboxes=bbox,
                time=time,
                filter=f'({label.osm_filter}) and ({filter_geometry})',
            ).as_dataframe()
            extent_data = gpd.GeoDataFrame(
                index=['extent'],
                crs='epsg:4326',
                geometry=[
                    shapely.geometry.box(*bbox, ccw=True),
                ],
            )

            vector_data['value'] = 1
            extent_data['value'] = 0
            vector_data = pd.concat([vector_data, extent_data], ignore_index=True)

            raster = self.__computer_raster(vector_data, utm)
            raster.rio.to_raster(raster_data)

        if raster_data.exists():
            with rasterio.open(raster_data) as dataset:
                data = dataset.read(1, out_shape=(dataset.count, height, width), resampling=Resampling.bilinear)

        return label_name, data

    @staticmethod
    def __calculate_id(text: str) -> uuid.UUID:
        hex_string = hashlib.md5(text.encode('UTF-8')).hexdigest()
        return uuid.UUID(hex=hex_string)

    def __computer_raster(self, vector_data: gpd.GeoDataFrame, utm: str) -> xarray.Dataset:
        """
        Sort the `vector_data` and then rasterise it with geocube, which takes the first value from the `vector_data`,
        ignoring duplicates if they exist.

        :param vector_data: a GeoDataFrame containing the queried OSM features (for one class), plus a 'default'
        geometry representing the full bounds, with a column `value` which is 1 for every feature and 0 for the
        'default' feature.
        :param utm: the utm code for the provided bbox
        """
        sorted_desc_areas_idx = vector_data.copy().to_crs(utm).geometry.area.argsort()[::-1]

        return make_geocube(
            vector_data=vector_data.iloc[sorted_desc_areas_idx],
            measurements=['value'],
            resolution=self.resolution,
            output_crs='EPSG:4326',
            fill=0,
        ).astype(np.uint8)

    @staticmethod
    def __utm_from_coords(area_coords: Tuple[float, float, float, float]) -> str:
        utm = query_utm_crs_info(datum_name='WGS 84', area_of_interest=AreaOfInterest(*area_coords))
        return f'{utm[0].auth_name}:{utm[0].code}'
