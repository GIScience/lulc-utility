from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from lulc.data.dataset import AreaDataset
from lulc.data.tx import MinMaxScaling, MaxScaling, Stack, ReclassifyMerge, ToTensor, NanToNum
from lulc.model.model import SegFormerModule
from coolname import generate_slug


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def train(cfg: DictConfig) -> None:
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

    model = SegFormerModule(num_channels=cfg.model.num_channels,
                            labels=dataset.labels,
                            variant=cfg.model.variant,
                            lr=cfg.model.lr,
                            device=device)

    train_loader = DataLoader(dataset, batch_size=cfg.model.batch_size)
    trainer = pl.Trainer(logger=neptune_logger,
                         max_epochs=cfg.model.max_epochs)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
