import torch
import torchvision.transforms.functional as tx_functional
from torchvision import transforms


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
        return {'x': tx_functional.crop(x, i, j, h, w), 'y': tx_functional.crop(y, i, j, h, w)}


class CenterCrop:
    def __init__(self, out_height=256, out_width=256):
        self.output_size = [out_height, out_width]

    def __call__(self, sample):
        x = sample['x']
        y = sample['y']
        return {
            'x': tx_functional.center_crop(x, self.output_size),
            'y': tx_functional.center_crop(y, self.output_size),
        }
