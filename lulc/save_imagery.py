import argparse
import logging
import os
import uuid
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
import yaml
from hydra import compose
from pyproj import CRS
from tqdm import tqdm

from lulc.ops.imagery_store_operator import resolve_imagery_store

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging/app/logging.yaml'
log = logging.getLogger(__name__)

config_dir = os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute()))

parser = argparse.ArgumentParser(prog='Save Imagery for Area', description='Save imagery for an area descriptor')
parser.add_argument(
    '--area-file', help='Provide an area descriptor csv file to override the area descriptor in the config files'
)


def cache_imagery(area_descriptor_file: str):
    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name='config')
    area_cfg = compose(config_name='area_descriptor')

    if not area_descriptor_file:
        area_descriptor_file = Path(area_cfg.area.output_dir) / f'area_{Path(area_cfg.area.aoi_file).stem}.csv'
    area_descriptor = pd.read_csv(area_descriptor_file)

    log.info(f'Configuring remote sensing imagery store: {cfg.imagery.operator}')
    imagery_store, _ = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))

    base_file_path = Path(f'./cache/imagery/{cfg.imagery.operator}/{uuid.uuid4()}')
    log.info(f'Saving images to {base_file_path}/')
    for i, tile in tqdm(area_descriptor.iterrows(), total=area_descriptor.shape[0]):
        tile_file_path = base_file_path / str(i)
        if not tile_file_path.exists():
            os.makedirs(tile_file_path)

        area_coords = tile[['min_x', 'min_y', 'max_x', 'max_y']].tolist()
        response = imagery_store.imagery(area_coords, '2022-06-01', '2024-06-01', resolution=cfg.imagery.resolution)

        for name, raw_img in response[0].items():
            if '.' in name:
                file_path = tile_file_path / name
            else:
                file_path = tile_file_path / f'{name}.tiff'

            if len(raw_img.shape) == 2:
                raw_img = np.expand_dims(raw_img, 2)

            img = np.transpose(raw_img, (2, 0, 1))

            with rasterio.open(
                file_path,
                mode='w',
                driver='GTiff',
                height=img.shape[1],
                width=img.shape[2],
                count=img.shape[0],
                dtype=img.dtype,
                crs=CRS.from_string('EPSG:4326'),
                transform=rasterio.transform.from_bounds(*area_coords, width=img.shape[2], height=img.shape[1]),
            ) as dst:
                dst.write(img)

    return


if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))

    args = parser.parse_args()
    cache_imagery(area_descriptor_file=args.area_file)
