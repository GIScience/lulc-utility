import logging.config
import os
from pathlib import Path
from shutil import rmtree

import hydra
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from lulc.data.collate import center_crop_collate_fn
from lulc.data.dataset import AreaDataset
from lulc.data.stats import dataset_iter_statistics
from lulc.ops.imagery_store_operator import resolve_imagery_store

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging/app/logging.yaml'
log = logging.getLogger(__name__)


# Set multiprocessing_context depending on system.#
#   Dataloader with default ('spawn') or with 'forkserver' is extremely slow on Linux (CPU) - much slower than
#   running just one worker).
#   https://stackoverflow.com/a/78343436/23656147
#   https://discuss.pytorch.org/t/data-loader-multiprocessing-slow-on-macos/131204/3
#
#   But 'fork' won't work with CUDA, so use 'forkserver' in that case (could potentially be 'spawn').
#   https://docs.pytorch.org/docs/stable/notes/multiprocessing.html
if torch.cuda.is_available():
    multiprocessing_context = 'forkserver'
else:
    multiprocessing_context = 'fork'


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

    log.info(f'Configuring remote sensing imagery store: {cfg.imagery.operator}')
    imagery_store, tr = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))

    dataset = AreaDataset(
        area_descriptor_ver=cfg.data.descriptor.area,
        label_descriptor_ver=cfg.data.descriptor.label,
        data_dir=Path(cfg.data.dir),
        cache_dir=Path(cfg.cache.dir),
        cache_items=cfg.cache.apply,
        imagery_store=imagery_store,
        resolution=cfg.imagery.resolution,
        deterministic_tx=transforms.Compose(tr),
    )

    if dataset.item_cache.exists():
        log.info('Dropping item intermediate cache (if exists)')
        rmtree(str(dataset.item_cache))

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.model.workers,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=False,
        collate_fn=center_crop_collate_fn(cfg.data.crop.height, cfg.data.crop.width),
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
