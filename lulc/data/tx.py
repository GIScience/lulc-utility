from abc import ABC
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms


class Tx(ABC):

    def __init__(self, subset='x'):
        self.subset = subset


class Stack(Tx):

    def __init__(self, subset='x'):
        super().__init__(subset)

    def __call__(self, sample):
        for layer, data in sample[self.subset].items():
            if len(data.shape) == 2:
                sample[self.subset][layer] = data[..., np.newaxis]

        sample[self.subset] = np.concatenate(list(sample[self.subset].values()), axis=-1)
        return sample


class ExtendShape(Tx):

    def __init__(self, subset='x'):
        super().__init__(subset)

    def __call__(self, sample):
        for subset_name, subset in sample.items():
            sample[subset_name] = subset[np.newaxis, ...]

        return sample


class ReclassifyMerge(Tx):

    def __init__(self, subset='y', method='denser_on_bottom'):
        super().__init__(subset)

        self.method = method

    def __call__(self, sample):
        layer_names = list(sample[self.subset].keys())

        if self.method == 'denser_on_bottom':
            layers = sorted(sample[self.subset].items(), key=lambda x: np.sum(x[1]), reverse=True)
        else:
            raise ValueError(f'Merge method {self.method} not supported')

        result = None
        for i, (layer_name, layer) in enumerate(layers):
            cls_idx = layer_names.index(layer_name) + 1

            if i == 0:
                result = layer * cls_idx
            else:
                result = np.where(layer == 1, layer * cls_idx, result)
            result = result.astype(np.int64)

        sample[self.subset] = result
        return sample


class ToTensor:

    def __call__(self, sample):
        for subset_name, subset in sample.items():
            tensor = torch.from_numpy(subset)
            if len(tensor.shape) == 3:
                tensor = tensor.permute(2, 0, 1)
            elif len(tensor.shape) == 4:
                tensor = tensor.permute(0, 3, 1, 2)

            sample[subset_name] = tensor

        return sample


class ToNumpy:

    def __call__(self, sample):
        for subset_name, subset in sample.items():
            sample[subset_name] = subset.numpy()

        return sample


class RandomCrop:

    def __init__(self, out_height=256, out_width=256):
        self.output_size = (out_height, out_width)

    def __call__(self, sample):
        x = sample['x']
        y = sample['y']
        i, j, h, w = transforms.RandomCrop.get_params(x, output_size=self.output_size)
        return {
            'x': F.crop(x, i, j, h, w),
            'y': F.crop(y, i, j, h, w)
        }


class CenterCrop:

    def __init__(self, out_height=256, out_width=256):
        self.output_size = [out_height, out_width]

    def __call__(self, sample):
        x = sample['x']
        y = sample['y']
        return {
            'x': F.center_crop(x, self.output_size),
            'y': F.center_crop(y, self.output_size)
        }


class NanToNum(Tx):

    def __init__(self, layers: List[str], subset='x', fill_value=0.0):
        super().__init__(subset)
        self.fill_value = fill_value
        self.layers = layers

    def __call__(self, sample):
        for layer in self.layers:
            tx_data: np.ndarray = np.nan_to_num(sample[self.subset][layer], nan=self.fill_value)

            sample[self.subset][layer] = tx_data
        return sample


class Normalize(Tx):
    def __init__(self, mean: List[float], std: List[float], subset='x'):
        super().__init__(subset)
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, sample):
        sample[self.subset] = self.normalize(sample[self.subset])
        return sample
