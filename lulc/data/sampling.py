from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from shapely import wkt
from torch.utils.data import Subset

from lulc.data.dataset import AreaDataset
import geopandas as gpd


class GeospatialStratifiedSampler:
    def __init__(self, dataset: AreaDataset, geometry_column: str = 'geometry'):
        self.dataset = dataset

        self.inner_descriptor = dataset.area_descriptor.copy()
        self.inner_descriptor[geometry_column] = self.inner_descriptor[geometry_column].apply(wkt.loads)
        self.inner_descriptor = gpd.GeoDataFrame(self.inner_descriptor, geometry=geometry_column)
        self.stratify_column = geometry_column

    def split_dataset(
        self, train_frac: float = 0.7, test_frac: float = 0.2, random_state: Optional[int] = None
    ) -> Tuple[Subset, Subset, Subset]:
        self.inner_descriptor['split_idx'] = self.inner_descriptor.groupby(by=self.stratify_column).ngroup()
        unique_col_values = self.inner_descriptor['split_idx'].unique()

        train_group_ids, val_group_ids = train_test_split(
            unique_col_values, test_size=(1 - train_frac), random_state=random_state
        )
        val_group_ids, test_group_ids = train_test_split(val_group_ids, test_size=test_frac, random_state=random_state)

        train_indices = self.inner_descriptor[self.inner_descriptor['split_idx'].isin(train_group_ids)].index.values
        val_indices = self.inner_descriptor[self.inner_descriptor['split_idx'].isin(val_group_ids)].index.values
        test_indices = self.inner_descriptor[self.inner_descriptor['split_idx'].isin(test_group_ids)].index.values

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)

        return train_dataset, val_dataset, test_dataset
