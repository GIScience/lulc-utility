import pytest
import torch

from lulc.model.model import SegFormerModule


@pytest.mark.parametrize(
    'variant, expected_num_params',
    [('MiT-b0', 3.7), ('MiT-b1', 14.0), ('MiT-b2', 25.4), ('MiT-b3', 45.2), ('MiT-b4', 62.6), ('MiT-b5', 82.0)],
)
def test_seg_former_module_variant(variant, expected_num_params):
    model = SegFormerModule(num_channels=5,
                            labels=['A', 'B'],
                            variant=variant,
                            lr=0.0001,
                            device=torch.device('cpu'))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6
    assert pytest.approx(num_params, 0.1) == expected_num_params


def test_step():
    model = SegFormerModule(num_channels=2,
                            labels=['A', 'B'],
                            variant='MiT-b0',
                            lr=0.0001,
                            device=torch.device('cpu'))

    for metric in model.metrics['test'].values():
        assert metric.compute() == torch.tensor(0) or torch.isnan(metric.compute())

    batch = {
        'x': torch.rand((2, 2, 256, 256)),
        'y': torch.randint(0, 2, (2, 256, 256))
    }

    loss = model._SegFormerModule__step(batch, 'test')
    assert loss != torch.nan

    for metric in model.metrics['test'].values():
        assert metric.compute() != torch.tensor(0) and not torch.isnan(metric.compute())
