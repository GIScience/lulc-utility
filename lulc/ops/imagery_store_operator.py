import logging
import logging as rio_logging
import shutil
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tarfile import ReadError
from typing import Callable, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from minio import Minio, S3Error
from omegaconf import DictConfig
from pyproj import Transformer
from rasterio.mask import mask
from rasterio.merge import merge
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
from sentinelhub.exceptions import SHRateLimitWarning

from lulc.data.tx.array import NanToNum, ReclassifyMerge, Stack, ToDtype
from lulc.ops.exception import OperatorInteractionError, OperatorValidationError

log = logging.getLogger(__name__)
rio_logging.getLogger('rasterio._filepath').setLevel(logging.ERROR)

# Upgrade SentinelHub rate limit warning to an error
warnings.filterwarnings('error', category=SHRateLimitWarning)


class ImageryStore(ABC):
    @abstractmethod
    def imagery(
        self,
        area_coords: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        resolution: int = 10,
    ) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
        pass

    def labels(
        self,
        area_coords: Tuple[float, float, float, float],
        end_date: str,
        resolution: int = 10,
    ) -> Tuple[np.ndarray, tuple[int, int]]:
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
        area_coords: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        resolution: int = 10,
    ) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        bbox = BBox(bbox=area_coords, crs=CRS.WGS84)
        bbox_width, bbox_height = bbox_to_dimensions(bbox, resolution=resolution)

        if bbox_width > 2500 or bbox_width > 2500:
            raise OperatorValidationError(
                'Area (after edge correction buffer) exceeds processing limit: 2500 px x 2500 px'
            )

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
            raise OperatorInteractionError(
                'SentinelHub operator interaction not possible. Please contact platform administrator.'
            )
        except ReadError:
            invalid_item = Path(request.get_filename_list()[0]).parent
            log.warning(f'Cached TAR file read error (cache id: {invalid_item}) - dropping and retrying download')

            shutil.rmtree(Path(request.data_folder) / invalid_item)
            return request.get_data(save_data=True)[0], (bbox_height, bbox_width)

    def labels(
        self,
        area_coords: Tuple[float, float, float, float],
        end_date: str,
        resolution: int = 10,
    ) -> Tuple[np.ndarray, tuple[int, int]]:
        """
        Get LULC classification from the Corine Land Cover inventory.
        """
        bbox = BBox(bbox=area_coords, crs=CRS.WGS84)
        bbox_width, bbox_height = bbox_to_dimensions(bbox, resolution=resolution)

        corine_years = self.corine_years

        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        end_date = list(filter(lambda year: year <= end_date.year, corine_years))[-1]
        end_date = f'{end_date}-01-01'

        if bbox_width > 2500 or bbox_height > 2500:
            raise OperatorValidationError(
                'Area (after edge correction buffer) exceeds processing limit: 2500 px x 2500 px'
            )

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
                    time_interval=(end_date, end_date),
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
            raise OperatorInteractionError(
                'Corine operator interaction not possible. Please contact platform administrator.'
            )


class FileOperator(ImageryStore):
    def __init__(self, tile_spec_path: Path, tile_dir: Path):
        """
        Initialise an imagery store for images saved in a local file directory.

        :param tile_dir: folder location containing tiled images
        :param tile_spec_path: path to parquet file describing the bounding boxes of the tiled images with an
        'id' column to resolve the filename (`tile_dir / f'{id}.tiff'`)
        """
        self.tile_dir = tile_dir
        self._load_tile_specification(path=Path(tile_spec_path).expanduser())

    def _load_tile_specification(self, path: Path) -> None:
        tile_specification = pd.read_parquet(path, engine='pyarrow')
        tile_specification['bbox'] = tile_specification['bbox'].apply(lambda b: shapely.Polygon.from_bounds(*b))
        tile_specification = gpd.GeoDataFrame(tile_specification, geometry='bbox')

        self.tile_specification = tile_specification
        self.sindex = self.tile_specification.sindex

        return

    def open_tiles(self, tiles: List[str]) -> List[rasterio.DatasetReader]:
        image_paths = [self.tile_dir / f'{qid}.tiff' for qid in tiles]

        sources = []
        for file in image_paths:
            if file.exists():
                src = rasterio.open(file)
                sources.append(src)

        return sources

    def imagery(
        self,
        area_coords: Tuple[float, float, float, float],
        start_date: str = '',
        end_date: str = '',
        resolution: int = 10,
    ) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
        return _create_image_from_tiles(
            area_coords=area_coords,
            tile_specification=self.tile_specification,
            sindex=self.sindex,
            tile_reader=self.open_tiles,
        )


class MinioOperator(ImageryStore):
    def __init__(self, minio_cfg: DictConfig, tile_spec_path: str, tile_dir: str):
        """
        Initialise an imagery store for images saved in MinIO.

        :param minio_cfg: configuration for the minio connection, including: host, port, access_key, secret_key, bucket
        :param tile_spec_path: the location of the tile specification parquet file within the bucket
        :param tile_dir: the location within the bucket containing the tiled images
        """
        # We don't initialise the client itself because that prevents us from deep copying objects containing this
        # imagery store, including from saving model hyperparameters if the dataset uses this imagery store.
        self.client_config = {
            'endpoint': f'{minio_cfg.host}:{minio_cfg.port}',
            'access_key': minio_cfg.access_key,
            'secret_key': minio_cfg.secret_key,
            'secure': True if minio_cfg.secure.lower() == 'true' else False,
        }
        self.bucket = minio_cfg.bucket
        self.tile_dir = tile_dir

        self._load_tile_specification(tile_spec_path)

    def _load_tile_specification(self, obj_path: str) -> None:
        if obj_path.startswith('file:'):
            tile_specification = pd.read_parquet(obj_path.replace('file:', ''), engine='pyarrow')
        else:
            client = Minio(**self.client_config)
            with client.get_object(bucket_name=self.bucket, object_name=obj_path) as response_stream:
                parquet_bytes = BytesIO(response_stream.read())
                tile_specification = pd.read_parquet(parquet_bytes)

        tile_specification['bbox'] = tile_specification['bbox'].apply(lambda b: shapely.Polygon.from_bounds(*b))
        self.tile_specification = gpd.GeoDataFrame(tile_specification, geometry='bbox')
        self.sindex = self.tile_specification.sindex

        return

    def open_tiles(self, tiles: List[str]) -> List[rasterio.DatasetReader]:
        client = Minio(**self.client_config)
        sources = []
        for tile in tiles:
            try:
                with client.get_object(
                    bucket_name=self.bucket, object_name=f'{self.tile_dir}/{tile}.tiff'
                ) as response_stream:
                    raster_bytes = BytesIO(response_stream.read())
                    src = rasterio.open(raster_bytes)
                    sources.append(src)
            except S3Error as exc:
                if exc.code != 'NoSuchKey':
                    log.warning('Error reading object from MinIO', exc_info=exc)
                    raise exc

        return sources

    def imagery(
        self,
        area_coords: Tuple[float, float, float, float],
        start_date: str = '',
        end_date: str = '',
        resolution: int = 10,
    ) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
        return _create_image_from_tiles(
            area_coords=area_coords,
            tile_specification=self.tile_specification,
            sindex=self.sindex,
            tile_reader=self.open_tiles,
        )


def _create_image_from_tiles(
    area_coords: Tuple[float, float, float, float],
    tile_specification: gpd.GeoDataFrame,
    sindex: gpd.sindex.BaseSpatialIndex,
    tile_reader: Callable,
) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    # Get tiles that cover the area_coords
    possible_matches_idx = list(sindex.intersection(area_coords))
    intersecting_tiles = tile_specification.iloc[possible_matches_idx]['id'].tolist()

    # Read tiles
    sources = tile_reader(intersecting_tiles)
    if not sources:
        raise RuntimeError(f'No valid sources found for {intersecting_tiles}')

    # Merge and clip tiles to the area_coords
    crs_transformer = Transformer.from_crs(4326, sources[0].crs)
    xmin, ymin = crs_transformer.transform(area_coords[1], area_coords[0])
    xmax, ymax = crs_transformer.transform(area_coords[3], area_coords[2])

    if len(sources) == 1:
        mosaic, _ = mask(
            dataset=sources[0],
            shapes=[shapely.Polygon.from_bounds(xmin, ymin, xmax, ymax)],
            all_touched=True,
            crop=True,
        )
    else:
        mosaic, _ = merge(sources, bounds=(xmin, ymin, xmax, ymax))

    # Rearrange the array and drop the "A" channel (from RGBA)
    mosaic = np.transpose(mosaic, (1, 2, 0))  # assuming it is (z, x, y) - transpose to (x, y, z)
    if mosaic.shape[2] == 4:
        vals = np.unique(mosaic[:, :, 3]).tolist()
        # Should actually only allow 255, but https://github.com/rasterio/rasterio/issues/2986
        assert set(vals) <= {0, 255}, (
            'The image has 4 channels. '
            f'Assuming RGBA format, the 4th channel should only contain 255, but it contains: {vals}'
        )
        mosaic = np.delete(mosaic, obj=3, axis=2)

    return {'rgb': mosaic}, mosaic.shape[:2]


def resolve_imagery_store(cfg: DictConfig, cache_dir: Path = None) -> ImageryStore:
    if cfg.operator == 'sentinel_hub':
        imagery_store = SentinelHubOperator(
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
        transforms = [
            NanToNum(layers=['s1.tif', 's2.tif']),
            Stack(),
            ReclassifyMerge(),
        ]

    elif cfg.operator in ('file_dir', 'minio'):
        transforms = [
            NanToNum(layers=['rgb']),
            Stack(),
            ReclassifyMerge(),
            ToDtype(dtype=np.float32),
        ]

        if cfg.operator == 'file_dir':
            imagery_store = FileOperator(
                tile_spec_path=Path(cfg.tile_spec_path).expanduser(), tile_dir=Path(cfg.tile_dir).expanduser()
            )
        else:
            imagery_store = MinioOperator(cfg.minio_platform, tile_spec_path=cfg.tile_spec_path, tile_dir=cfg.tile_dir)

    else:
        raise ValueError(f'Cannot resolve imagery operator {cfg.operator}')

    return imagery_store, transforms
