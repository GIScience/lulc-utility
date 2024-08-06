import copy
import logging
from functools import partial
from typing import Dict

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lulc.data.augment import build_random_tx
from lulc.data.collate import center_crop_collate_fn, random_crop_collate_fn
from lulc.data.dataset import AreaDataset

from lulc.data.sampling import GeospatialStratifiedSampler

log = logging.getLogger(__name__)


class AreaDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: AreaDataset,
        batch_size: int,
        num_workers: int,
        crop_height: int,
        crop_width: int,
        train_frac: float,
        test_frac: float,
        augment: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_height = crop_height
        self.crop_width = crop_width

        train_size = int(train_frac * len(dataset))
        test_size = int(test_frac * len(dataset))
        val_size = len(dataset) - (train_size + test_size)

        log.info(f'Performing stratified dataset split (train: {train_size}, val: {val_size, }test: {test_size})')
        sampler = GeospatialStratifiedSampler(dataset, 'geometry')

        self.train_dataset, self.val_dataset, self.test_dataset = sampler.split_dataset()

        log.info(f'Attaching random transformations to the training dataset: {list(augment.keys())}')
        self.train_dataset.dataset = copy.deepcopy(self.train_dataset.dataset)
        self.train_dataset.dataset.random_tx = build_random_tx(augment)

        self.loader_p = partial(
            DataLoader,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.loader_p(
            dataset=self.train_dataset,
            shuffle=True,
            collate_fn=random_crop_collate_fn(self.crop_height, self.crop_width),
        )

    def val_dataloader(self) -> DataLoader:
        return self.loader_p(
            dataset=self.val_dataset, collate_fn=center_crop_collate_fn(self.crop_height, self.crop_width)
        )

    def test_dataloader(self) -> DataLoader:
        return self.loader_p(
            dataset=self.test_dataset, collate_fn=center_crop_collate_fn(self.crop_height, self.crop_width)
        )
