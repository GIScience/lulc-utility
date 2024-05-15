import hashlib
import io
import logging
import logging as rio_logging
import shutil
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tarfile import ReadError
from typing import Dict, List, Tuple

import ee
import numpy as np
import rasterio as rio
import requests
from ee import EEException
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


class EarthEngineOperator(ImageryStore):
    def __init__(
        self,
        service_account: str,
        key_file_location: Path,
        s1_bands: List[str],
        s2_bands: List[str],
        dem_bands: List[str],
        cache_dir: Path,
        cloud_filter=60,
        cloud_probability_threshold=40,
        nir_dark_regions_threshold=0.15,
        cloud_projection_distance=2,
        cloud_dilation_buffer=100,
        sr_band_scale=1e4,
    ):
        credentials = ee.ServiceAccountCredentials(service_account, str(key_file_location))
        ee.Initialize(credentials)

        self.s2_bands = s2_bands
        self.s1_bands = s1_bands
        self.dem_bands = dem_bands
        self.cache_dir: Path = cache_dir / 'ee'
        self.s1_cache_dir = self.cache_dir / 's1'
        self.s2_cache_dir = self.cache_dir / 's2'
        self.dem_cache_dir = self.cache_dir / 'dem'
        self.cloud_filter = cloud_filter
        self.cloud_probability_threshold = cloud_probability_threshold
        self.nir_dark_regions_threshold = nir_dark_regions_threshold
        self.cloud_projection_distance = cloud_projection_distance
        self.cloud_dilation_buffer = cloud_dilation_buffer
        self.sr_band_scale = sr_band_scale

        for path in [self.s1_cache_dir, self.s2_cache_dir, self.dem_cache_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def imagery(
        self,
        area_coords: Tuple,
        start_date: str,
        end_date: str,
        resolution: int = 10,
    ) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        area_of_interest = ee.Geometry.BBox(*area_coords)
        hex_string = hashlib.md5('_'.join([str(x) for x in area_coords]).encode('UTF-8')).hexdigest()
        item_id = uuid.UUID(hex=hex_string)

        s1_file = self.s1_cache_dir / f'{item_id}.tiff'
        s2_file = self.s2_cache_dir / f'{item_id}.tiff'
        dem_file = self.dem_cache_dir / f'{item_id}.tiff'

        def read_raster(file):
            with rio.open(str(file), mode='r') as dataset:
                return np.transpose(dataset.read(), [1, 2, 0])

        try:
            s1 = (
                read_raster(s1_file)
                if s1_file.exists()
                else self.__download_s1_data(s1_file, area_of_interest, start_date, end_date, resolution)
            )
            s2 = (
                read_raster(s2_file)
                if s2_file.exists()
                else self.__download_cloudless_s2_data(s2_file, area_of_interest, start_date, end_date, resolution)
            )
            dem = (
                read_raster(dem_file)
                if dem_file.exists()
                else self.__download_dem(dem_file, area_of_interest, resolution)
            )

            return {
                's1.tif': s1.astype(np.float32),
                's2.tif': s2.astype(np.float32),
                'dem.tif': dem.astype(np.float32),
            }, dem.shape[:2]

        except EEException:
            log.exception('Data download not possible')
            raise OperatorInteractionException(
                'Earth Engine operator interaction not possible. Please contact platform administrator.'
            )

    def __download_s1_data(self, output_file: Path, area_of_interest, start_date, end_date, resolution):
        s1_grd_col = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(area_of_interest)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filterDate(start_date, end_date)
        )
        ee_image = s1_grd_col.median()

        return self.__ee_image_to_numpy(output_file, ee_image, area_of_interest, resolution, self.s1_bands)

    def __download_cloudless_s2_data(
        self,
        output_file: Path,
        area_of_interest: ee.Geometry.BBox,
        start_date: str,
        end_date: str,
        resolution: int,
    ):
        def apply_cld_shdw_mask(img):
            not_cld_shdw = img.select('cloudmask').Not()
            return img.select('B.*').updateMask(not_cld_shdw)

        def add_cloud_bands(img):
            cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
            is_cloud = cld_prb.gt(self.cloud_probability_threshold).rename('clouds')
            return img.addBands(ee.Image([cld_prb, is_cloud]))

        def add_shadow_bands(img):
            not_water = img.select('SCL').neq(6)
            dark_pixels = (
                img.select('B8')
                .lt(self.nir_dark_regions_threshold * self.sr_band_scale)
                .multiply(not_water)
                .rename('dark_pixels')
            )
            shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
            cloud_projection = (
                img.select('clouds')
                .directionalDistanceTransform(shadow_azimuth, self.cloud_projection_distance * 10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform')
            )

            shadows = cloud_projection.multiply(dark_pixels).rename('shadows')
            return img.addBands(ee.Image([dark_pixels, cloud_projection, shadows]))

        def add_cld_shdw_mask(img):
            img_cloud = add_cloud_bands(img)
            img_cloud_shadow = add_shadow_bands(img_cloud)
            is_cloud_shadow = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
            is_cloud_shadow = (
                is_cloud_shadow.focalMin(2)
                .focalMax(self.cloud_dilation_buffer * 2 / 20)
                .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                .rename('cloudmask')
            )
            return img_cloud_shadow.addBands(is_cloud_shadow)

        s2_sr_col = (
            ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(area_of_interest)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_filter))
        )

        s2_cloudless_col = (
            ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(area_of_interest)
            .filterDate(start_date, end_date)
        )

        s2_collection = ee.ImageCollection(
            ee.Join.saveFirst('s2cloudless').apply(
                **{
                    'primary': s2_sr_col,
                    'secondary': s2_cloudless_col,
                    'condition': ee.Filter.equals(**{'leftField': 'system:index', 'rightField': 'system:index'}),
                }
            )
        )

        ee_image = (s2_collection.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median(),)
        return self.__ee_image_to_numpy(output_file, ee_image, area_of_interest, resolution, self.s2_bands)

    def __download_dem(self, output_file: Path, area_of_interest: ee.Geometry.BBox, resolution: int):
        dem_collection = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM')
        ee_image = dem_collection.mosaic()
        return self.__ee_image_to_numpy(output_file, ee_image, area_of_interest, resolution, self.dem_bands)

    @staticmethod
    def __ee_image_to_numpy(
        output_file: Path, ee_image: ee.Image, area_of_interest: ee.Geometry.BBox, resolution: int, bands: List[str]
    ):
        def aux(band: str) -> rio.DatasetReader:
            download_url = ee_image.getDownloadUrl(
                {
                    'format': 'GEO_TIFF',
                    'bands': [band],
                    'region': area_of_interest,
                    'scale': resolution,
                }
            )
            response = requests.get(download_url)
            return rio.open(io.BytesIO(response.content), mode='r', driver='GTiff')

        with ThreadPoolExecutor() as pool:
            datasets_r = list(pool.map(aux, bands))

            profile = datasets_r[0].profile
            profile.update(
                dtype=rio.int16,
                count=len(bands),
                compress='lzw',
            )

            with rio.open(str(output_file), 'w+', **profile) as target_dataset:
                for i, dataset in enumerate(datasets_r, start=1):
                    target_dataset.write(dataset.read(1), i)

                return target_dataset.read()


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
    elif cfg.operator == 'earth_engine':
        return EarthEngineOperator(
            cfg.service_account,
            cfg.key_file_location,
            cfg.s1_bands,
            cfg.s2_bands,
            cfg.dem_bands,
            cache_dir=cache_dir / 'imagery',
        )
    else:
        raise ValueError(f'Cannot resolve imagery operator {cfg.operator}')
