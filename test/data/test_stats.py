import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset, DataLoader

from lulc.data.stats import dataset_iter_statistics

data = [{
    'x': Uniform(0, 1).sample(torch.Size([3, 256, 256])),
    'y': torch.randint(0, 4, (256, 256))
}, {
    'x': Uniform(0, 1).sample(torch.Size([3, 256, 256])),
    'y': torch.randint(0, 4, (256, 256))
}, {
    'x': Uniform(0, 1).sample(torch.Size([3, 256, 256])),
    'y': torch.randint(0, 4, (256, 256))
}]
stacked_data_x = torch.stack([item['x'] for item in data])
stacked_data_y = torch.stack([item['y'] for item in data]).flatten().numpy()


class TestDataset(Dataset):

    def __len__(self):
        return len(data)

    def __getitem__(self, item):
        return data[item]


def test_dataset_statistics():
    expected_mean = torch.mean(stacked_data_x, dim=[0, 2, 3])
    expected_std = torch.std(stacked_data_x, dim=[0, 2, 3])
    expected_class_weight = compute_class_weight('balanced', classes=np.unique(stacked_data_y), y=stacked_data_y)

    loader = DataLoader(TestDataset(),
                        batch_size=2,
                        num_workers=0,
                        persistent_workers=False)
    statistics = dataset_iter_statistics(loader, labels=['background', 'agriculture', 'water', 'built-up'])
    assert torch.allclose(statistics.mean, expected_mean, atol=0.0001)
    assert torch.allclose(statistics.std, expected_std, atol=0.0001)
    assert np.allclose(statistics.class_weights, expected_class_weight, atol=0.0001)
