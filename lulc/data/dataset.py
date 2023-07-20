from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from lulc.ops.osm_operator import OhsomeOps
from lulc.ops.sentinelhub_operator import SentinelHubOperator


class AreaDataset(Dataset):

    def __init__(self, area_descriptor_ver: str,
                 label_descriptor_ver: str,
                 imagery_descriptor_ver: str,
                 sentinelhub_api_id: str,
                 sentinelhub_secret: str,
                 data_dir: Path,
                 cache_dir: Path,
                 deterministic_tx: transforms.Compose,
                 random_tx: Optional[transforms.Compose] = None
                 ):
        self.osm = OhsomeOps(cache_dir=cache_dir / 'osm')
        self.sentinelhub = SentinelHubOperator(api_id=sentinelhub_api_id,
                                               api_secret=sentinelhub_secret,
                                               evalscript_dir=data_dir / 'imagery',
                                               evalscript_name=f'imagery_{imagery_descriptor_ver}',
                                               cache_dir=cache_dir / 'sentinelhub')
        self.area_descriptor = pd.read_csv(str(data_dir / 'area' / f'area_{area_descriptor_ver}.csv'))

        label_descriptor = pd.read_csv(str(data_dir / 'label' / f'label_{label_descriptor_ver}.csv'),
                                       index_col='label')
        self.labels, self.osm_lulc_mapping = self.__interpret_label_descriptor(label_descriptor)

        self.item_cache = cache_dir / 'items' / area_descriptor_ver / label_descriptor_ver / imagery_descriptor_ver
        self.deterministic_tx = deterministic_tx
        self.random_tx = random_tx

    def __len__(self):
        return len(self.area_descriptor)

    def __getitem__(self, idx):
        area = self.area_descriptor.iloc[idx]
        area_coords = tuple(area[['min_x', 'min_y', 'max_x', 'max_y']].values)
        imagery, imagery_size = self.sentinelhub.imagery(area_coords, area['start_date'], area['end_date'])
        labels = self.osm.labels(area_coords, area['end_date'], self.osm_lulc_mapping, imagery_size)
        item = {
            'x': imagery,
            'y': labels
        }

        item_path = self.item_cache / str(idx)
        x_path = f'{item_path}/x.pt'
        y_path = f'{item_path}/y.pt'
        if not item_path.exists():
            item_path.mkdir(parents=True)
            item = self.deterministic_tx(item)
            torch.save(item['x'], x_path)
            torch.save(item['y'], y_path)
        else:
            item = {
                'x': torch.load(x_path),
                'y': torch.load(y_path)
            }

        return item if self.random_tx is None else self.random_tx(item)

    @staticmethod
    def __interpret_label_descriptor(descriptor: pd.DataFrame) -> (List[str], Dict):
        osm_lulc = dict((str(k), v['filter']) for k, v in descriptor.iterrows())
        return ['unidentified'] + list(osm_lulc.keys()), osm_lulc
