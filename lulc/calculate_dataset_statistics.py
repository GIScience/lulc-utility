import logging
from pathlib import Path
from shutil import rmtree

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from data.collate import center_crop_collate_fn
from lulc.data.dataset import AreaDataset
from lulc.data.stats import dataset_iter_statistics
from lulc.data.tx.array import Stack, ReclassifyMerge, NanToNum
from lulc.data.tx.tensor import ToTensor
from lulc.ops.imagery_store_operator import resolve_imagery_store

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def calculate_dataset_statistics(cfg: DictConfig) -> None:
    """

    Iterate through the dataset to calculate statistics needed for standardization / normalization
    procedure.
    :param cfg: underlying Hydra configuration
    :return: mean, std and class weights
    """
    log.info(f'Configuring remote sensing imagery store: {cfg.imagery.operator}')
    imagery_store = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))

    dataset = AreaDataset(
        area_descriptor_ver=cfg.data.descriptor.area,
        label_descriptor_ver=cfg.data.descriptor.labels,
        data_dir=Path(cfg.data.dir),
        cache_dir=Path(cfg.cache.dir),
        imagery_store=imagery_store,
        deterministic_tx=transforms.Compose([
            NanToNum(layers=['s1.tif', 's2.tif']),
            Stack(),
            ReclassifyMerge(),
            ToTensor()
        ])
    )

    if dataset.item_cache.exists():
        log.info('Dropping item intermediate cache (if exists)')
        rmtree(str(dataset.item_cache))

    loader = DataLoader(dataset,
                        batch_size=cfg.model.batch_size,
                        num_workers=0,
                        persistent_workers=False,
                        collate_fn=center_crop_collate_fn(cfg.data.crop.height, cfg.data.crop.width))

    statistics = dataset_iter_statistics(loader, dataset.labels)
    log.info(f'Channel mean: {statistics.mean.tolist()}')
    log.info(f'Channel std: {statistics.std.tolist()}')
    log.info(f'Class weights: {statistics.class_weights.tolist()}')


if __name__ == "__main__":
    calculate_dataset_statistics()
