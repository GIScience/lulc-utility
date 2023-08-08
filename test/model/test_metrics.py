import torch

from lulc.model.metrics import ConfusionMatrix2D


def test_confusion_matrix_when_preds_equal_targets():
    cm = ConfusionMatrix2D(labels=['A', 'B', 'C'], task='multiclass', num_classes=3, normalize='true')
    input = torch.tensor([[[1, 2, 0],
                           [0, 1, 0],
                           [2, 2, 1]]])

    cm.update(input, input)
    result = cm.compute()
    assert torch.equal(result, torch.tensor([[1.0, .0, .0],
                                             [.0, 1.0, .0],
                                             [.0, .0, 1.0]]))


def test_confusion_matrix():
    cm = ConfusionMatrix2D(labels=['A', 'B', 'C'], task='multiclass', num_classes=3, normalize='true')

    preds = torch.tensor([[[1, 2, 0],
                           [0, 1, 0],
                           [2, 2, 1]]])
    target = torch.tensor([[[0, 2, 0],
                            [0, 1, 0],
                            [0, 2, 0]]])
    cm.update(preds, target)
    result = cm.compute()
    assert torch.allclose(result, torch.tensor([[0.5000, 0.3333, 0.1667],
                                                [0.0000, 1.0000, 0.0000],
                                                [0.0000, 0.0000, 1.0000]]), atol=0.0001)

def test_confusion_matrix_reset():
    cm = ConfusionMatrix2D(labels=['A', 'B', 'C'], task='multiclass', num_classes=3, normalize='true')

    preds = torch.randint(0, 2, (2, 1024, 1024))
    target = torch.randint(0, 2, (2, 1024, 1024))
    cm.update(preds, target)
    assert not torch.all(cm.compute() == 0)
    cm.reset()
    assert torch.all(cm.compute() == 0)
