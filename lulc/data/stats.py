import math
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class DatasetStatistics:
    mean: torch.Tensor
    std: torch.Tensor
    class_weights: torch.Tensor


def dataset_iter_statistics(loader: DataLoader, labels: List[str]) -> DatasetStatistics:
    """
    Iteratively calculates dataset statistics to avoid storing large datasets in memory.

    :param loader: Torch loader that contains all or sampled training dataset.
    :param labels: Classes used by the prediction mechanism.
    :return:
    """

    n_samples = len(loader.dataset)

    x_ch_mean = 0
    x_ch_sq_mean = 0
    y_bincount = torch.zeros(len(labels))

    for data in tqdm(loader, total=len(loader)):
        x, y = data['x'], data['y']
        frac = y.shape[0] / n_samples

        x_ch_mean += torch.mean(x, dim=[0, 2, 3]) * frac
        x_ch_sq_mean += torch.mean(x**2, dim=[0, 2, 3]) * frac

        y_samples = math.prod(y.shape)
        y_bincount += (torch.bincount(torch.flatten(y), minlength=len(labels)) / y_samples) * frac

    x_ch_std = (x_ch_sq_mean - x_ch_mean**2) ** 0.5
    class_weights = 1 / (len(labels) * y_bincount)

    return DatasetStatistics(x_ch_mean, x_ch_std, class_weights)
