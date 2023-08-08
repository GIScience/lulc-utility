from torch.utils.data import default_collate

from lulc.data.tx.tensor import RandomCrop, CenterCrop


def random_crop_collate_fn(crop_height: int, crop_width):
    crop = RandomCrop(crop_height, crop_width)

    def random_crop_collate(batch):
        batch = [crop(sample) for sample in batch]
        return default_collate(batch)

    return random_crop_collate


def center_crop_collate_fn(crop_height: int, crop_width: int):
    crop = CenterCrop(crop_height, crop_width)

    def random_crop_collate(batch):
        batch = [crop(sample) for sample in batch]
        return default_collate(batch)

    return random_crop_collate
