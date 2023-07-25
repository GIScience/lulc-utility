import logging
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from lulc.data.dataset import AreaDataset
from lulc.data.tx import MinMaxScaling, MaxScaling, Stack, ReclassifyMerge, ToTensor, NanToNum
from lulc.model.model import SegFormerModule
from coolname import generate_slug

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def train(cfg: DictConfig) -> None:
    log.info('Model training initiated')

    torch.set_float32_matmul_precision(cfg.model.matmul_precision)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    neptune_logger = NeptuneLogger(
        name=generate_slug(pattern=3),
        project=cfg.neptune.project,
        api_key=cfg.neptune.api_token,
        log_model_checkpoints=False,
        mode=cfg.neptune.mode
    )
    neptune_logger.log_hyperparams(params=cfg.model)

    log.info(f'Initializing dataset (area: {cfg.data.descriptor.area}, '
             f'label: {cfg.data.descriptor.labels}, imagery: {cfg.data.descriptor.imagery})')

    dataset = AreaDataset(
        area_descriptor_ver=cfg.data.descriptor.area,
        label_descriptor_ver=cfg.data.descriptor.labels,
        imagery_descriptor_ver=cfg.data.descriptor.imagery,
        sentinelhub_api_id=cfg.sentinel_hub.api.id,
        sentinelhub_secret=cfg.sentinel_hub.api.secret,
        data_dir=Path(cfg.data.dir),
        cache_dir=Path(cfg.cache.dir),
        deterministic_tx=transforms.Compose([
            NanToNum(layers=['s1.tif', 's2.tif']),
            MinMaxScaling(layers=['s1.tif', 'dem.tif']),
            MaxScaling(layers=['s2.tif']),
            Stack(),
            ReclassifyMerge(),
            ToTensor(ch_first=True)
        ])
    )

    train_size = int(cfg.data.train_frac * len(dataset))
    test_size = int(cfg.data.test_frac * len(dataset))
    val_size = len(dataset) - (train_size + test_size)

    log.info(f'Performing random dataset split (train: {train_size}, val: {val_size}, test: {test_size})')
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.model.batch_size,
                              pin_memory=True,
                              num_workers=cfg.model.workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.model.batch_size,
                            pin_memory=True,
                            num_workers=cfg.model.workers)
    test_loader = DataLoader(val_dataset,
                             batch_size=cfg.model.batch_size,
                             pin_memory=True,
                             num_workers=cfg.model.workers)

    log.info(f'Creating a model ({cfg.model.variant})')
    model = SegFormerModule(num_channels=cfg.model.num_channels,
                            labels=dataset.labels,
                            variant=cfg.model.variant,
                            lr=cfg.model.lr,
                            device=device)

    log.info(f'Training model for {cfg.model.max_epochs} epochs')
    trainer = pl.Trainer(logger=neptune_logger, max_epochs=cfg.model.max_epochs)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log.info('Performing model evaluation')
    # TODO implement in trainer
    trainer.test(model=model, dataloaders=test_loader)

    log.info('Model training completed')


if __name__ == "__main__":
    train()
