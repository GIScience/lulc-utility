import numpy as np
import pytest
import torch

from lulc.data.tx import Stack, ReclassifyMerge, ToTensor, RandomCrop, CenterCrop, Normalize

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

test_data_c = {
    'x': torch.randint(0, 255, (9, 256, 256), dtype=torch.float32),
    'y': torch.randint(0, 5, (256, 256))
}

test_data_d = {
    'x': torch.cat([
        torch.full((1, 256, 256), 0.5, dtype=torch.float32),
        torch.full((1, 256, 256), 10, dtype=torch.float32),
        torch.full((1, 256, 256), 200, dtype=torch.float32)
    ]),
    'y': torch.randint(0, 5, (256, 256))
}


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
    tx = ToTensor()
    result = tx(test_data_b)
    assert (9, 256, 256) == result['x'].shape
    assert torch.float64 == result['x'].dtype
    assert (256, 256) == result['y'].shape
    assert torch.int64 == result['y'].dtype


def test_random_crop():
    tx = RandomCrop(out_height=64, out_width=64)
    result_a = tx(test_data_c)
    result_b = tx(test_data_c)
    assert not torch.equal(result_a['x'], result_b['x'])
    assert not torch.equal(result_a['y'], result_b['y'])


def test_random_crop_when_crop_smaller_than_input():
    tx = RandomCrop(out_height=128, out_width=128)
    result = tx(test_data_c)
    assert (9, 128, 128) == result['x'].shape
    assert (128, 128) == result['y'].shape


def test_random_crop_when_crop_larger_than_input():
    tx = RandomCrop(out_height=512, out_width=512)

    with pytest.raises(ValueError):
        tx(test_data_c)


def test_center_crop():
    tx = CenterCrop(out_height=64, out_width=64)
    result_a = tx(test_data_c)
    result_b = tx(test_data_c)
    assert (9, 64, 64) == result_a['x'].shape
    assert (64, 64) == result_a['y'].shape

    assert torch.equal(result_a['x'], result_b['x'])
    assert torch.equal(result_a['y'], result_b['y'])


def test_normalize():
    tx = Normalize(mean=[0.0, 1.0, 400.0], std=[1.0, 1.0, 100])
    result = tx(test_data_d)
    assert (3, 256, 256) == result['x'].shape
    assert (256, 256) == result['y'].shape
