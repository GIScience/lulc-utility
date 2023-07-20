import numpy as np
import torch

from lulc.data.tx import MinMaxScaling, MaxScaling, Stack, ReclassifyMerge, ToTensor

test_data_a = {
    'x': {
        's1.tif': np.random.uniform(0, 2000, (256, 256, 2)),
        's2.tif': np.random.randint(0, 255, (256, 256, 6)),
        'dem.tif': np.random.uniform(-5, 500, (256, 256)),
    },
    'y': {
        'forest': np.random.randint(0, 2, (256, 256)),
        'water': np.zeros((256, 256), dtype=np.int8),
        'urban': np.random.randint(0, 2, (256, 256)),
        'built-up': np.random.randint(0, 2, (256, 256))
    }
}

test_data_b = {
    'x': np.random.uniform(0, 255, (256, 256, 9)),
    'y': np.random.randint(0, 5, (256, 256))
}


def test_normalize_reflectance():
    tx = MaxScaling(layers=['s2.tif'])
    result: np.ndarray = tx(test_data_a)['x']['s2.tif']
    assert (256, 256, 6) == result.shape
    assert np.float32 == result.dtype


def test_normalize_dem():
    tx = MinMaxScaling(layers=['dem.tif', 's1.tif'])
    result: np.ndarray = tx(test_data_a)['x']['dem.tif']
    assert (256, 256) == result.shape
    assert np.float32 == result.dtype

    result: np.ndarray = tx(test_data_a)['x']['s1.tif']
    assert (256, 256, 2) == result.shape
    assert np.float32 == result.dtype


def test_stack():
    tx = Stack(subset='x')
    result: np.ndarray = tx(test_data_a)['x']
    assert (256, 256, 9) == result.shape


def test_merge():
    tx = ReclassifyMerge(subset='y')
    result: np.ndarray = tx(test_data_a)['y']
    assert (256, 256) == result.shape
    assert 0 == result.min()
    assert 4 == result.max()


def test_to_tensor():
    tx = ToTensor(ch_first=True)
    result = tx(test_data_b)
    assert (9, 256, 256) == result['x'].shape
    assert torch.float64 == result['x'].dtype
    assert (256, 256) == result['y'].shape
    assert torch.int64 == result['y'].dtype
