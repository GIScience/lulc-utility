import logging
from functools import partial
from pathlib import Path
from shutil import rmtree

import hydra
import lightning.pytorch as pl
import torch
from coolname import generate_slug
from lightning import seed_everything
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from lulc.data.augment import build_random_tx
from lulc.data.collate import random_crop_collate_fn, center_crop_collate_fn
from lulc.data.dataset import AreaDataset
from lulc.data.tx.array import Normalize, Stack, ReclassifyMerge, NanToNum
from lulc.model.model import SegformerModule
from model.ops.registry import NeptuneModelRegistry
from ops.imagery_store_operator import resolve_imagery_store
from monitoring.energy import EnergyContext

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def train(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.model.matmul_precision)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = generate_slug(pattern=3)
    output_dir = HydraConfig.get().runtime.output_dir

    if cfg.environment.deterministic:
        seed_everything(42, workers=True)
        torch.use_deterministic_algorithms(True, warn_only=True)

    log.info(f'Model training initiated: {run_name}')
    neptune_logger = NeptuneLogger(
        name=run_name,
        project=cfg.neptune.project,
        api_key=cfg.neptune.api_token,
        log_model_checkpoints=False,
        mode=cfg.neptune.mode,
        prefix=''
    )
    neptune_logger.log_hyperparams(params=cfg.model)
    neptune_logger.experiment['data/area'] = cfg.data.descriptor.area
    neptune_logger.experiment['data/label'] = cfg.data.descriptor.labels
    neptune_logger.experiment['data/imagery'] = cfg.imagery.operator

    log.info(f'Configuring remote sensing imagery store: {cfg.imagery.operator}')
    imagery_store = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))

    with EnergyContext(neptune_logger.experiment) as energy_context:

        log.info(f'Initializing dataset (area: {cfg.data.descriptor.area}, label: {cfg.data.descriptor.labels})')

        dataset = AreaDataset(
            area_descriptor_ver=cfg.data.descriptor.area,
            label_descriptor_ver=cfg.data.descriptor.labels,
            imagery_store=imagery_store,
            data_dir=Path(cfg.data.dir),
            cache_dir=Path(cfg.cache.dir),
            deterministic_tx=transforms.Compose([
                NanToNum(layers=['s1.tif', 's2.tif']),
                Stack(),
                ReclassifyMerge(),
                Normalize(mean=cfg.data.normalize.mean, std=cfg.data.normalize.std)
            ])
        )

        if dataset.item_cache.exists():
            log.info('Refreshing item intermediate cache (if exists)')
            rmtree(str(dataset.item_cache))

        train_size = int(cfg.data.train_frac * len(dataset))
        test_size = int(cfg.data.test_frac * len(dataset))
        val_size = len(dataset) - (train_size + test_size)

        log.info(f'Performing random dataset split (train: {train_size}, val: {val_size}, test: {test_size})')
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        log.info(f'Attaching random transformations to the training dataset: {list(cfg.model.augment.keys())}')
        train_dataset.dataset.random_tx = build_random_tx(cfg.model.augment)

        log.info('Configuring data loaders')
        loader_p = partial(DataLoader, batch_size=cfg.model.batch_size,
                           pin_memory=True,
                           num_workers=cfg.model.workers,
                           persistent_workers=True)
        train_loader = loader_p(dataset=train_dataset, shuffle=True, collate_fn=random_crop_collate_fn(cfg.data.crop.height, cfg.data.crop.width))
        val_loader = loader_p(dataset=val_dataset, collate_fn=center_crop_collate_fn(cfg.data.crop.height, cfg.data.crop.width))
        test_loader = loader_p(dataset=test_dataset, collate_fn=center_crop_collate_fn(cfg.data.crop.height, cfg.data.crop.width))

        log.info(f'Creating a model ({cfg.model.variant})')
        params = dict(
            num_channels=cfg.model.num_channels,
            labels=dataset.labels,
            variant=cfg.model.variant,
            lr=cfg.model.lr,
            device=device,
            class_weights=cfg.data.class_weights,
            color_codes=dataset.color_codes,
            max_image_samples=cfg.model.max_image_samples
        )
        model = SegformerModule(**params)
        log.info(f'Training model for {cfg.model.max_epochs} epochs')
        energy_context.record('train')

        early_stopping = EarlyStopping(monitor='val/epoch/loss', patience=cfg.model.early_stopping.patience)
        model_checkpoint = ModelCheckpoint(dirpath=f'{output_dir}/checkpoints', save_top_k=2, monitor='val/epoch/loss')

        trainer = pl.Trainer(logger=neptune_logger,
                             max_epochs=cfg.model.max_epochs,
                             callbacks=[early_stopping, model_checkpoint],
                             log_every_n_steps=cfg.model.log_every_n_steps,
                             gradient_clip_val=cfg.model.gradient_clip_val,
                             deterministic=cfg.environment.deterministic,
                             profiler=cfg.environment.profiler)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        log.info(f'Checking-out best model: {model_checkpoint.best_model_path}')
        model = SegformerModule.load_from_checkpoint(model_checkpoint.best_model_path, **params)

        log.info('Performing model evaluation')
        energy_context.record('test')
        trainer.test(model=model, dataloaders=test_loader)

        log.info(f'Model training completed: {run_name}')
        registry = NeptuneModelRegistry(model_key=cfg.neptune.model.key,
                                        project=cfg.neptune.project,
                                        api_key=cfg.neptune.api_token,
                                        cache_dir=Path(cfg.cache.dir))
        energy_context.record('register')
        registry.register_version(model, run_name, neptune_logger.experiment.get_url())


if __name__ == "__main__":
    train()
