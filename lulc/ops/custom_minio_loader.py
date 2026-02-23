import logging
import os
from io import BytesIO
from pathlib import Path
from typing import List

import geopandas as gpd
import hydra
import rasterio
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from minio import Minio, S3Error

from lulc.ops.imagery_store_operator import MinioOperator

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging.yaml'
log = logging.getLogger(__name__)

tile_obj_dir = 'by_country/{code}/global_quarterly_2024q2_mosaic'


def read_tile_grouping(
    tile_grouping_path: str = 'by_country/aoi.geojson', group_id_col: str = 'NAME'
) -> gpd.GeoDataFrame:
    if not GlobalHydra().is_initialized():
        config_dir = os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute()))
        hydra.initialize_config_dir(config_dir=config_dir, version_base=None)

    cfg = compose(config_name='config')
    minio_cfg = cfg.train.imagery['minio_platform']
    client_config = {
        'endpoint': f'{minio_cfg["host"]}:{minio_cfg["port"]}',
        'access_key': minio_cfg['access_key'],
        'secret_key': minio_cfg['secret_key'],
        'secure': True if minio_cfg['secure'].lower() == 'true' else False,
    }

    client = Minio(**client_config)
    with client.get_object(bucket_name=minio_cfg['bucket'], object_name=tile_grouping_path) as response_stream:
        tile_grouping = gpd.read_file(response_stream)
        tile_grouping['code'] = tile_grouping[group_id_col].str.lower()

    return tile_grouping[['code', 'geometry']]


MINIO_TILE_GROUPING = read_tile_grouping()


def open_tiles_by_country(minio_operator: MinioOperator, tiles: gpd.GeoDataFrame) -> List[rasterio.DatasetReader]:
    client = Minio(**minio_operator.client_config)
    sources = []
    for _, tile in tiles.iterrows():
        possible_country_group = MINIO_TILE_GROUPING.to_crs(tiles.crs).intersects(tile['bbox'])

        for _, group in MINIO_TILE_GROUPING[possible_country_group].iterrows():
            tile_dir = tile_obj_dir.format(code=group['code'])

            try:
                with client.get_object(
                    bucket_name=minio_operator.bucket, object_name=f'{tile_dir}/{tile.id}.tiff'
                ) as response_stream:
                    raster_bytes = BytesIO(response_stream.read())
                    src = rasterio.open(raster_bytes)
                    sources.append(src)

                break
            except S3Error as exc:
                if exc.code != 'NoSuchKey':
                    log.warning('Error reading object from MinIO', exc_info=exc)
                    raise exc

    return sources
