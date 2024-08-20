import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from lulc.data.label import resolve_osm_labels
from lulc.data.tx.tensor import ToTensor
from lulc.ops.imagery_store_operator import ImageryStore
from lulc.ops.osm_operator import OhsomeOps


class AreaDataset(Dataset):
    def __init__(
        self,
        area_descriptor_ver: str,
        label_descriptor_ver: str,
        imagery_store: ImageryStore,
        data_dir: Path,
        cache_dir: Path,
        deterministic_tx: transforms.Compose,
        random_tx: Optional[transforms.Compose] = None,
        cache_items: bool = True,
    ):
        self.osm = OhsomeOps(cache_dir=cache_dir / 'osm' / label_descriptor_ver)
        self.imagery_store = imagery_store

        self.area_descriptor = pd.read_csv(str(data_dir / 'area' / f'area_{area_descriptor_ver}.csv'))

        label_descriptors = resolve_osm_labels(data_dir, label_descriptor_ver)
        self.labels = [d.name for d in label_descriptors]
        self.osm_lulc_mapping = dict([(d.name, d) for d in label_descriptors if d.osm_filter is not None])
        self.color_codes = [d.color for d in label_descriptors]

        self.item_cache = cache_dir / 'items' / area_descriptor_ver / label_descriptor_ver
        self.cache_items = cache_items
        self.deterministic_tx = deterministic_tx
        self.random_tx = random_tx

    def __len__(self):
        return len(self.area_descriptor)

    def __getitem__(self, idx):
        if self.cache_items:
            item_path = self.item_cache / str(idx)
            x_path = f'{item_path}/x.pt'
            y_path = f'{item_path}/y.pt'

            if not item_path.exists() or len(os.listdir(item_path)) == 0:
                item = self.__compute_item(idx)
                item_path.mkdir(parents=True, exist_ok=True)
                torch.save(item['x'], x_path)
                torch.save(item['y'], y_path)
            else:
                item = {'x': torch.load(x_path), 'y': torch.load(y_path)}
        else:
            item = self.__compute_item(idx)

        item = item if self.random_tx is None else self.random_tx(item)
        return ToTensor()(item)

    def __compute_item(self, idx):
        area = self.area_descriptor.iloc[idx]
        area_coords = tuple(area[['min_x', 'min_y', 'max_x', 'max_y']].values)
        imagery, imagery_size = self.imagery_store.imagery(area_coords, area['start_date'], area['end_date'])
        labels = self.osm.labels(area_coords, area['end_date'], self.osm_lulc_mapping, imagery_size)
        return self.deterministic_tx({'x': imagery, 'y': labels})
