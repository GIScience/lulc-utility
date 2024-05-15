import numpy as np
import pytest
import torch

from lulc.data.tx.tensor import CenterCrop, RandomCrop, ToTensor

test_data_channel_last = {
    'x': np.random.uniform(0, 255, (256, 256, 9)),
    'y': np.random.randint(0, 5, (256, 256)),
}

test_data_channel_first = {
    'x': torch.randint(0, 255, (9, 256, 256), dtype=torch.float32),
    'y': torch.randint(0, 5, (256, 256)),
}


def test_to_tensor():
    tx = ToTensor()
    result = tx(test_data_channel_last)
    assert (9, 256, 256) == result['x'].shape
    assert torch.float64 == result['x'].dtype
    assert (256, 256) == result['y'].shape
    assert torch.int64 == result['y'].dtype


def test_random_crop():
    tx = RandomCrop(out_height=64, out_width=64)
    result_a = tx(test_data_channel_first)
    result_b = tx(test_data_channel_first)
    assert not torch.equal(result_a['x'], result_b['x'])
    assert not torch.equal(result_a['y'], result_b['y'])


def test_random_crop_when_crop_smaller_than_input():
    tx = RandomCrop(out_height=128, out_width=128)
    result = tx(test_data_channel_first)
    assert (9, 128, 128) == result['x'].shape
    assert (128, 128) == result['y'].shape


def test_random_crop_when_crop_larger_than_input():
    tx = RandomCrop(out_height=512, out_width=512)

    with pytest.raises(ValueError):
        tx(test_data_channel_first)


def test_center_crop():
    tx = CenterCrop(out_height=64, out_width=64)
    result_a = tx(test_data_channel_first)
    result_b = tx(test_data_channel_first)
    assert (9, 64, 64) == result_a['x'].shape
    assert (64, 64) == result_a['y'].shape

    assert torch.equal(result_a['x'], result_b['x'])
    assert torch.equal(result_a['y'], result_b['y'])
