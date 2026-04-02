import logging
import os
import uuid
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
import yaml
from omegaconf import DictConfig
from pyproj import CRS
from tqdm import tqdm

from lulc.data.area import extract_area_name
from lulc.ops.imagery_store_operator import resolve_imagery_store

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging.yaml'
log = logging.getLogger(__name__)

config_dir = os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute()))


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def cache_imagery(cfg: DictConfig) -> None:
    area_name = extract_area_name(cfg.train.area)
    area_descriptor = pd.read_csv(str(Path(cfg.train.area.output_dir) / f'{area_name}.csv'))

    log.info(f'Configuring remote sensing imagery store: {cfg.train.imagery.operator}')
    imagery_store, _ = resolve_imagery_store(cfg.train.imagery, cache_dir=Path(cfg.cache.dir))

    base_file_path = Path(f'./cache/imagery/{cfg.train.imagery.operator}/{uuid.uuid4()}')
    log.info(f'Saving images to {base_file_path}/')
    for i, tile in tqdm(area_descriptor.iterrows(), total=area_descriptor.shape[0]):
        tile_file_path = base_file_path / str(i)
        if not tile_file_path.exists():
            os.makedirs(tile_file_path)

        area_coords = tile[['min_x', 'min_y', 'max_x', 'max_y']].tolist()
        response = imagery_store.imagery(
            area_coords, '2022-06-01', '2024-06-01', resolution=cfg.train.imagery.resolution
        )

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
    cache_imagery()
