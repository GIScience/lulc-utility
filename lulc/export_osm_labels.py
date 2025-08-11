"""
Note: this is currently largely duplicated with `lulc/calculate_dataset_statistics.py`, but hydra and argparse don't
play ball very well, so for now we are just accepting the duplication.
"""

import logging.config
import os
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig
from torchvision import transforms

from lulc.data.dataset import AreaDataset
from lulc.ops.imagery_store_operator import resolve_imagery_store

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging/app/logging.yaml'
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def export_osm_labels(cfg: DictConfig) -> None:
    """
    Iterate through the dataset and export the ground truth labels from the OSM filters.

    :param cfg: underlying Hydra configuration
    :return: mean, std and class weights
    """
    log.info(f'Configuring remote sensing imagery store: {cfg.imagery.operator}')
    imagery_store, tr = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))

    log.info('Initialising area dataset')
    dataset = AreaDataset(
        area_descriptor_ver=cfg.data.descriptor.area,
        label_descriptor_ver=cfg.data.descriptor.label,
        imagery_store=imagery_store,
        resolution=cfg.imagery.resolution,
        data_dir=Path(cfg.data.dir),
        cache_dir=Path(cfg.cache.dir),
        deterministic_tx=transforms.Compose(tr),
        cache_items=cfg.cache.apply,
    )

    dataset.export_osm_labels()


if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))
    log.info('Visualising labels')
    export_osm_labels()
