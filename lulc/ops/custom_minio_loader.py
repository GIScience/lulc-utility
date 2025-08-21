import logging
import os
from io import BytesIO
from typing import List

import geopandas as gpd
import rasterio
from hydra import compose
from minio import Minio, S3Error

from lulc.ops.imagery_store_operator import MinioOperator

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging.yaml'
log = logging.getLogger(__name__)

tile_obj_dir = 'by_country/{code}/global_quarterly_2024q2_mosaic'


def read_tile_grouping(
    tile_grouping_path: str = 'by_country/aoi.geojson', group_id_col: str = 'NAME'
) -> gpd.GeoDataFrame:
    # Hydra config is already loaded, so we just need to connect to it here
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


tile_grouping = read_tile_grouping()


def open_tiles(minio_operator: MinioOperator, tiles: gpd.GeoDataFrame) -> List[rasterio.DatasetReader]:
    client = Minio(**minio_operator.client_config)
    sources = []
    for _, tile in tiles.iterrows():
        possible_country_group = tile_grouping.to_crs(tiles.crs).intersects(tile['bbox'])

        for _, group in tile_grouping[possible_country_group].iterrows():
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
