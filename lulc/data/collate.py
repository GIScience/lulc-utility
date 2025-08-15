from functools import partial

from torch.utils.data import default_collate

from lulc.data.tx.tensor import CenterCrop, RandomCrop


def random_crop_collate_fn(crop_height: int, crop_width):
    return partial(random_crop_collate, crop_height, crop_width)


def random_crop_collate(crop_height, crop_width, batch):
    crop = RandomCrop(crop_height, crop_width)
    batch = [crop(sample) for sample in batch]
    return default_collate(batch)


def center_crop_collate_fn(crop_height: int, crop_width: int):
    return partial(center_crop_collate, crop_height, crop_width)


def center_crop_collate(crop_height, crop_width, batch):
    crop = CenterCrop(crop_height, crop_width)
    batch = [crop(sample) for sample in batch]
    return default_collate(batch)
