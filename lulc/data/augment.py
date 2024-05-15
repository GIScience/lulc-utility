from typing import Dict

import numpy as np
from albumentations import CoarseDropout, Compose, ElasticTransform, HorizontalFlip, Rotate, VerticalFlip

AUGMENTATIONS = {
    'random_rotate': Rotate,
    'horizontal_flip': HorizontalFlip,
    'vertical_flip': VerticalFlip,
    'elastic_transform': ElasticTransform,
    'coarse_dropout': CoarseDropout,
}


class ComposeWrapper:
    def __init__(self, compose: Compose):
        self.compose = compose

    def __call__(self, sample):
        augmentation = self.compose(image=sample['x'], mask=sample['y'])
        return {
            'x': augmentation['image'].astype(np.float32),
            'y': augmentation['mask'].astype(np.int64),
        }


def build_random_tx(cfg: Dict):
    compose = Compose([AUGMENTATIONS[augmentation_name](**params) for augmentation_name, params in cfg.items()])
    return ComposeWrapper(compose)
