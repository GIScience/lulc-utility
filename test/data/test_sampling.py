from unittest.mock import Mock

import numpy as np
import pandas as pd
from shapely import wkt

from lulc.data.sampling import GeospatialStratifiedSampler


def test_sampling():
    dataset = Mock()
    dataset.area_descriptor = pd.DataFrame(
        data={
            'geometry': [
                'POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))',
                'POLYGON ((30 10, 40 10, 40 20, 30 20, 30 10))',
                'POLYGON ((50 10, 60 10, 60 20, 50 20, 50 10))',
                'POLYGON ((70 10, 80 10, 80 20, 70 20, 70 10))',
                'POLYGON ((90 10, 100 10, 100 20, 90 20, 90 10))',
                'POLYGON ((10 30, 20 30, 20 40, 10 40, 10 30))',
                'POLYGON ((30 30, 40 30, 40 40, 30 40, 30 30))',
                'POLYGON ((50 30, 60 30, 60 40, 50 40, 50 30))',
                'POLYGON ((70 30, 80 30, 80 40, 70 40, 70 30))',
                'POLYGON ((90 30, 100 30, 100 40, 90 40, 90 30))',
                'POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))',
                'POLYGON ((30 10, 40 10, 40 20, 30 20, 30 10))',
                'POLYGON ((50 10, 60 10, 60 20, 50 20, 50 10))',
                'POLYGON ((70 10, 80 10, 80 20, 70 20, 70 10))',
                'POLYGON ((90 10, 100 10, 100 20, 90 20, 90 10))',
                'POLYGON ((10 30, 20 30, 20 40, 10 40, 10 30))',
                'POLYGON ((30 30, 40 30, 40 40, 30 40, 30 30))',
                'POLYGON ((50 30, 60 30, 60 40, 50 40, 50 30))',
                'POLYGON ((70 30, 80 30, 80 40, 70 40, 70 30))',
                'POLYGON ((90 30, 100 30, 100 40, 90 40, 90 30))',
            ]
        }
    )

    sampler = GeospatialStratifiedSampler(dataset=dataset)

    train_dataset, val_dataset, test_dataset = sampler.split_dataset(random_state=42)

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    combined_indices = set(np.concatenate((train_indices, val_indices, test_indices)))
    dataset_indices = set(np.arange(len(dataset.area_descriptor)))

    np.testing.assert_equal(combined_indices, dataset_indices, err_msg='Whole dataset is not split')

    np.testing.assert_equal(train_indices, np.array([2, 3, 4, 6, 7, 9, 12, 13, 14, 16, 17, 19]))
    np.testing.assert_equal(val_indices, np.array([0, 5, 8, 10, 15, 18]))
    np.testing.assert_equal(test_indices, np.array([1, 11]))

    train_geometries = set(dataset.area_descriptor.iloc[train_indices]['geometry'].apply(wkt.loads))
    val_geometries = set(dataset.area_descriptor.iloc[val_indices]['geometry'].apply(wkt.loads))
    test_geometries = set(dataset.area_descriptor.iloc[test_indices]['geometry'].apply(wkt.loads))

    assert train_geometries.isdisjoint(val_geometries), 'Train and Validation geometries overlap.'
    assert train_geometries.isdisjoint(test_geometries), 'Train and Test geometries overlap.'
    assert val_geometries.isdisjoint(test_geometries), 'Validation and Test geometries overlap.'

    assert set(list(train_indices) + list(val_indices) + list(test_indices)) == set(range(len(dataset.area_descriptor)))
