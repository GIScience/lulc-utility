import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import rasterio
import sentinelhub
import torch
from rasterio import CRS
from rasterio.merge import merge
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from lulc.data.label import resolve_osm_labels
from lulc.data.tx.tensor import ToTensor
from lulc.ops.imagery_store_operator import ImageryStore
from lulc.ops.osm_operator import OhsomeOps

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging/app/logging.yaml'
log = logging.getLogger(__name__)


class AreaDataset(Dataset):
    def __init__(
        self,
        area_descriptor_ver: str,
        label_descriptor_ver: str,
        imagery_store: ImageryStore,
        resolution: int,
        data_dir: Path,
        cache_dir: Path,
        deterministic_tx: transforms.Compose,
        random_tx: Optional[transforms.Compose] = None,
        cache_items: bool = True,
    ):
        """
        A torch dataset class to get image samples and their ground truth labels for the machine learning processes.

        Also includes the following public methods:
        - `export_osm_labels`: to generate a raster of each class (a.k.a. label) for the full area defined
          in `imagery_store.area_coords` and save this for visual inspection
        """
        self.osm = OhsomeOps(cache_dir=cache_dir / 'osm' / label_descriptor_ver / area_descriptor_ver)
        self.imagery_store = imagery_store
        self.resolution = resolution

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
        """
        Get the image and corresponding rasterised (OSM) labels and apply the transforms from `self.deterministic_tx`.
        """
        area = self.area_descriptor.iloc[idx]
        area_coords = tuple(area[['min_x', 'min_y', 'max_x', 'max_y']].values)
        imagery, imagery_size = self.imagery_store.imagery(area_coords, area['start_date'], area['end_date'])
        labels = self.osm.labels(area_coords, area['end_date'], self.osm_lulc_mapping, imagery_size)

        return self.deterministic_tx({'x': imagery, 'y': labels})

    def export_osm_labels(self):
        """
        Get the OSM labels for every sub-area and then save a merged raster of the ground truth labels.
        """
        for _, area in tqdm(self.area_descriptor.iterrows(), total=self.area_descriptor.shape[0]):
            area_coords = tuple(area[['min_x', 'min_y', 'max_x', 'max_y']].values)
            bbox = sentinelhub.BBox(bbox=area_coords, crs=sentinelhub.CRS.WGS84)
            imagery_size = sentinelhub.bbox_to_dimensions(bbox, resolution=self.resolution)

            _ = self.osm.labels(area_coords, area['end_date'], self.osm_lulc_mapping, imagery_size)

        log.info('Exporting rasters of generated osm labels')
        self._merge_osm_rasters()
        return

    def _merge_osm_rasters(self):
        for label in os.listdir(self.osm.cache_dir):
            if (self.osm.cache_dir / label).is_file():
                continue

            paths = os.listdir(self.osm.cache_dir / label)
            raster, transform = merge([self.osm.cache_dir / label / p for p in paths if p.endswith('.tiff')])

            with rasterio.open(
                self.osm.cache_dir / f'{label}.tiff',
                mode='w',
                driver='GTiff',
                height=raster.shape[1],
                width=raster.shape[2],
                count=raster.shape[0],
                dtype=raster.dtype,
                crs=CRS.from_string('EPSG:4326'),
                transform=transform,
                nodata=0,
            ) as dst:
                dst.write(raster)
