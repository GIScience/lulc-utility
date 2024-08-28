import pytest
import torch

from lulc.model.model import SegformerModule


@pytest.mark.parametrize(
    'variant, expected_num_params',
    [('MiT-b0', 3.7), ('MiT-b1', 14.0), ('MiT-b2', 25.4), ('MiT-b3', 45.2), ('MiT-b4', 62.6), ('MiT-b5', 82.0)],
)
def test_seg_former_module_variant(variant, expected_num_params):
    model = SegformerModule(
        num_channels=5,
        labels=['A', 'B'],
        variant=variant,
        lr=0.0001,
        device=torch.device('cpu'),
        class_weights=[1, 0.5],
        color_codes=[[0, 0, 0], [255, 0, 0]],
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6
    assert pytest.approx(num_params, 0.1) == expected_num_params


def test_step():
    model = SegformerModule(
        num_channels=2,
        labels=['A', 'B'],
        variant='MiT-b0',
        lr=0.0001,
        device=torch.device('cpu'),
        class_weights=[1, 0.5],
        color_codes=[[0, 0, 0], [255, 0, 0]],
    )

    for metric in model.metrics['test'].values():
        assert torch.all(metric.compute() == 0) or torch.all(torch.isnan(metric.compute()))

    batch = {'x': torch.rand((2, 2, 256, 256)), 'y': torch.randint(0, 2, (2, 256, 256))}

    loss = model.step(batch, 'test')
    assert loss != torch.nan

    for metric in model.metrics['test'].values():
        assert torch.all(metric.compute().sum() != 0)
        assert not torch.all(torch.isnan(metric.compute()))


@pytest.mark.parametrize(
    'input_shape, expected_shape',
    [
        ((2, 3, 256, 256), (2, 256, 256, 3)),
        ((2, 7, 256, 256), (2, 256, 256, 3)),
        ((2, 5, 256, 256), (2, 256, 256, 3)),
        ((2, 1, 256, 256), (2, 256, 256, 3)),
    ],
)
def test_image_grid(input_shape, expected_shape):
    x = torch.rand(input_shape)
    result = SegformerModule.image_grid(x)
    assert result.shape == torch.Size(expected_shape)
