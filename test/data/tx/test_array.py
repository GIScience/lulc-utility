import numpy as np

from lulc.data.tx.array import Normalize, ReclassifyMerge, Stack

test_data_random = {
    'x': {
        's1.tif': np.random.uniform(0, 2000, (256, 256, 2)),
        's2.tif': np.random.randint(0, 255, (256, 256, 6)),
        'dem.tif': np.random.uniform(-5, 500, (256, 256)),
    },
    'y': {
        'forest': np.random.randint(0, 2, (256, 256)),
        'water': np.zeros((256, 256), dtype=np.int8),
        'urban': np.random.randint(0, 2, (256, 256)),
        'built-up': np.random.randint(0, 2, (256, 256)),
    },
}

test_data_deterministic_x = {
    'x': np.concatenate(
        [
            np.full((256, 256, 1), 0.5, dtype=np.float32),
            np.full((256, 256, 1), 10, dtype=np.float32),
            np.full((256, 256, 1), 200, dtype=np.float32),
        ],
        axis=-1,
    ),
    'y': np.random.randint(0, 5, (256, 256)),
}


def test_stack():
    tx = Stack(subset='x')
    result: np.ndarray = tx(test_data_random)['x']
    assert (256, 256, 9) == result.shape


def test_merge():
    tx = ReclassifyMerge(subset='y')
    result: np.ndarray = tx(test_data_random)['y']
    assert (256, 256) == result.shape
    assert 0 == result.min()
    assert 4 == result.max()


def test_normalize():
    tx = Normalize(mean=[0.0, 1.0, 400.0], std=[1.0, 1.0, 100])
    result = tx(test_data_deterministic_x)
    assert (256, 256, 3) == result['x'].shape
    assert (256, 256) == result['y'].shape
