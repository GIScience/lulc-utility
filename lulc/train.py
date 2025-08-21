import logging.config
import os
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
import yaml
from coolname import generate_slug
from hydra.core.hydra_config import HydraConfig
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.tuner import Tuner
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from lulc.data.dataset import AreaDataset
from lulc.data.module import AreaDataModule
from lulc.data.tx.array import Normalize
from lulc.model.model import SegformerModule
from lulc.model.ops.registry import NeptuneModelRegistry
from lulc.monitoring.energy import EnergyContext
from lulc.ops.imagery_store_operator import resolve_imagery_store

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging.yaml'
log = logging.getLogger(__name__)
plt.switch_backend('agg')


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def train(cfg: DictConfig) -> None:
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision(cfg.train.model.matmul_precision)
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
        mode=cfg.train.model.neptune.mode,
        prefix='',
    )
    area_descriptor = getattr(cfg.train.area, 'aoi_file', getattr(cfg.train.area, 'aoi_name', None))
    neptune_logger.log_hyperparams(params=cfg.train.model)
    neptune_logger.experiment['data/area'] = area_descriptor
    neptune_logger.experiment['data/label'] = cfg.train.label.descriptor
    neptune_logger.experiment['data/imagery'] = cfg.train.imagery.operator

    log.info(f'Configuring remote sensing imagery store: {cfg.train.imagery.operator}')
    imagery_store, tr = resolve_imagery_store(cfg.train.imagery, cache_dir=Path(cfg.cache.dir))

    with EnergyContext(neptune_logger.experiment, enable_tracking=cfg.environment.energy_tracker) as energy_context:
        log.info(f'Initializing dataset (area: {area_descriptor}, label: {cfg.train.label.descriptor})')

        dataset = AreaDataset(
            area_cfg=cfg.train.area,
            label_filter=cfg.train.label,
            imagery_store=imagery_store,
            resolution=cfg.train.imagery.resolution,
            data_dir=Path(cfg.train.data.dir),
            cache_dir=Path(cfg.cache.dir),
            cache_items=cfg.cache.apply,
            deterministic_tx=transforms.Compose(
                tr
                + [
                    Normalize(mean=cfg.train.data.normalize.mean, std=cfg.train.data.normalize.std),
                ]
            ),
        )

        datamodule = AreaDataModule(
            dataset=dataset,
            batch_size=cfg.train.model.batch_size,
            num_workers=cfg.train.model.workers,
            crop_height=cfg.train.data.crop.height,
            crop_width=cfg.train.data.crop.width,
            train_frac=cfg.train.data.train_frac,
            test_frac=cfg.train.data.test_frac,
            augment=OmegaConf.to_container(cfg.train.model.augment, resolve=True),
        )

        log.info(f'Creating a model ({cfg.train.model.variant})')
        params = dict(
            num_channels=cfg.train.model.num_channels,
            labels=dataset.labels,
            variant=cfg.train.model.variant,
            lr=cfg.train.model.lr,
            device=device,
            class_weights=cfg.train.data.class_weights,
            color_codes=dataset.color_codes,
            max_image_samples=cfg.train.model.max_image_samples,
            temperature=cfg.train.model.temperature,
            label_smoothing=cfg.train.model.label_smoothing,
        )
        model = SegformerModule(**params)

        log.info(
            f'Training model for {"unlimited" if cfg.train.model.max_epochs == -1 else cfg.train.model.max_epochs} epochs'
        )
        energy_context.record('train')

        early_stopping = EarlyStopping(monitor='val/epoch/loss', patience=cfg.train.model.early_stopping.patience)
        model_checkpoint = ModelCheckpoint(dirpath=f'{output_dir}/checkpoints', save_top_k=2, monitor='val/epoch/loss')

        trainer = pl.Trainer(
            logger=neptune_logger,
            max_epochs=cfg.train.model.max_epochs,
            callbacks=[early_stopping, model_checkpoint],
            log_every_n_steps=cfg.train.model.log_every_n_steps,
            gradient_clip_val=cfg.train.model.gradient_clip_val,
            deterministic=cfg.environment.deterministic,
            profiler=cfg.environment.profiler,
        )

        if cfg.train.model.enable_tuning:
            tuner = Tuner(trainer)
            tuner.lr_find(model, datamodule=datamodule)
            tuner.scale_batch_size(model, mode='binsearch', datamodule=datamodule, init_val=cfg.train.model.batch_size)

        trainer.fit(model=model, datamodule=datamodule)

        log.info(f'Checking-out best model: {model_checkpoint.best_model_path}')
        model = SegformerModule.load_from_checkpoint(model_checkpoint.best_model_path, **params)

        log.info('Performing model evaluation')
        energy_context.record('test')
        trainer.test(model=model, datamodule=datamodule)

        log.info(f'Model training completed: {run_name}')
        registry = NeptuneModelRegistry(
            model_key=cfg.train.model.neptune.model_key,
            project=cfg.neptune.project,
            api_key=cfg.neptune.api_token,
            cache_dir=Path(cfg.cache.dir),
        )
        energy_context.record('register')
        registry.register_version(
            model=model,
            run_name=run_name,
            run_url=neptune_logger.experiment.get_url(),
            label_descriptor_version=cfg.train.label.descriptor,
        )


if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))
    log.info('Starting model training')
    train()
