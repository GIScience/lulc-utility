import logging.config
import os
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from lulc.data.collate import center_crop_collate_fn
from lulc.data.dataset import AreaDataset
from lulc.data.module import MULTIPROCESSING_CONTEXT
from lulc.data.stats import dataset_iter_statistics
from lulc.ops.imagery_store_operator import resolve_imagery_store

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging.yaml'
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def calculate_dataset_statistics(cfg: DictConfig) -> None:
    """
    Iterate through the imagery dataset to calculate statistics needed for standardization / normalization
    procedure.

    Saves the dataset statistics for training to `conf/data/`.

    :param cfg: underlying Hydra configuration
    :return: mean, std and class weights
    """
    torch.multiprocessing.set_start_method('spawn')

    log.info(f'Configuring remote sensing imagery store: {cfg.train.imagery.operator}')
    imagery_store, tr = resolve_imagery_store(cfg.train.imagery, cache_dir=Path(cfg.cache.dir))

    dataset = AreaDataset(
        area_cfg=cfg.train.area,
        label_filter=cfg.train.label,
        data_dir=Path(cfg.train.data.dir),
        cache_dir=Path(cfg.cache.dir),
        cache_items=cfg.cache.apply,
        imagery_store=imagery_store,
        resolution=cfg.train.imagery.resolution,
        deterministic_tx=transforms.Compose(tr),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.train.model.workers,
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
        persistent_workers=False,
        collate_fn=center_crop_collate_fn(cfg.train.data.crop.height, cfg.train.data.crop.width),
    )

    log.info('Loading images and iteratively calculating dataset statistics')
    statistics = dataset_iter_statistics(loader, dataset.labels)
    log.info(f'Channel mean: {statistics.mean.tolist()}')
    log.info(f'Channel std: {statistics.std.tolist()}')
    log.info(f'Class weights: {statistics.class_weights.tolist()}')


if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))
    log.info('Calculating dataset statistics')
    calculate_dataset_statistics()
