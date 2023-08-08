import numpy as np
from omegaconf import DictConfig

from lulc.data.augment import build_random_tx


def test_build_random_tx():
    cfg = DictConfig({
        'horizontal_flip': {
            'p': 1
        }
    })
    tx = build_random_tx(cfg)
    item = tx({
        'x': np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]),
        'y': np.array([
            [1, 2, 3],
            [0, 1, 2],
            [0, 0, 1],
        ]),
    })

    assert np.array_equal(item['x'], np.array([
        [3, 2, 1],
        [6, 5, 4],
        [9, 8, 7],
    ]))
    assert np.array_equal(item['y'], np.array([
        [3, 2, 1],
        [2, 1, 0],
        [1, 0, 0],
    ]))
