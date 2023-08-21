import os
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from lulc.data.label import resolve_labels
from lulc.data.tx.tensor import ToTensor
from lulc.ops.imagery_store_operator import ImageryStore
from lulc.ops.osm_operator import OhsomeOps


class AreaDataset(Dataset):

    def __init__(self, area_descriptor_ver: str,
                 label_descriptor_ver: str,
                 imagery_store: ImageryStore,
                 data_dir: Path,
                 cache_dir: Path,
                 deterministic_tx: transforms.Compose,
                 random_tx: Optional[transforms.Compose] = None
                 ):
        self.osm = OhsomeOps(cache_dir=cache_dir / 'osm' / label_descriptor_ver)
        self.imagery_store = imagery_store

        self.area_descriptor = pd.read_csv(str(data_dir / 'area' / f'area_{area_descriptor_ver}.csv'))

        label_descriptor = pd.read_csv(str(data_dir / 'label' / f'label_{label_descriptor_ver}.csv'),
                                       index_col='label')
        self.labels, self.osm_lulc_mapping = self.__interpret_label_descriptor(label_descriptor)

        self.item_cache = cache_dir / 'items' / area_descriptor_ver / label_descriptor_ver
        self.deterministic_tx = deterministic_tx
        self.random_tx = random_tx
        self.color_codes = resolve_labels(data_dir, label_descriptor_ver).color_codes

    def __len__(self):
        return len(self.area_descriptor)

    def __getitem__(self, idx):
        item_path = self.item_cache / str(idx)
        x_path = f'{item_path}/x.pt'
        y_path = f'{item_path}/y.pt'

        if not item_path.exists() or len(os.listdir(item_path)) == 0:
            area = self.area_descriptor.iloc[idx]
            area_coords = tuple(area[['min_x', 'min_y', 'max_x', 'max_y']].values)
            imagery, imagery_size = self.imagery_store.imagery(area_coords, area['start_date'], area['end_date'])
            labels = self.osm.labels(area_coords, area['end_date'], self.osm_lulc_mapping, imagery_size)

            item = self.deterministic_tx({
                'x': imagery,
                'y': labels
            })

            item_path.mkdir(parents=True, exist_ok=True)
            torch.save(item['x'], x_path)
            torch.save(item['y'], y_path)
        else:
            item = {
                'x': torch.load(x_path),
                'y': torch.load(y_path)
            }

        item = item if self.random_tx is None else self.random_tx(item)
        return ToTensor()(item)

    @staticmethod
    def __interpret_label_descriptor(descriptor: pd.DataFrame) -> (List[str], Dict):
        osm_lulc = dict((str(k), v['filter']) for k, v in descriptor.iterrows())
        return ['unidentified'] + list(osm_lulc.keys()), osm_lulc
