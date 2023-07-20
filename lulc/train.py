from pathlib import Path

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from lulc.data.dataset import AreaDataset
from lulc.data.tx import MinMaxScaling, MaxScaling, Stack, ReclassifyMerge, ToTensor, NanToNum
from lulc.model.model import SegFormerModule


@hydra.main(version_base=None, config_path='../conf', config_name='config')
def train(cfg: DictConfig) -> None:
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

    model = SegFormerModule(num_channels=cfg.model.num_channels, labels=dataset.labels)
    train_loader = DataLoader(dataset)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train()
