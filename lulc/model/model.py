from typing import List, Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import MeanMetric, JaccardIndex, Accuracy, F1Score, Precision, Recall
from transformers import SegformerForSemanticSegmentation, \
    SegformerConfig

MODEL_VARIANTS = {
    'MiT-b0': {
        'depths': [2, 2, 2, 2],
        'hidden_sizes': [32, 64, 160, 256],
        'decoder_hidden_size': 256
    },
    'MiT-b1': {
        'depths': [2, 2, 2, 2],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 256
    },
    'MiT-b2': {
        'depths': [3, 4, 6, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768
    },
    'MiT-b3': {
        'depths': [3, 4, 18, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768
    },
    'MiT-b4': {
        'depths': [3, 8, 27, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768
    },
    'MiT-b5': {
        'depths': [3, 6, 40, 3],
        'hidden_sizes': [64, 128, 320, 512],
        'decoder_hidden_size': 768
    }
}


class SegformerModule(pl.LightningModule):

    def __init__(self, num_channels: int, labels: List[str], variant: str, lr: float, device: torch.device,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        num_labels = len(labels)
        label2id = {k: v for v, k in enumerate(labels)}
        id2label = {v: k for k, v in label2id.items()}

        variant_config = MODEL_VARIANTS[variant]
        self.configuration = SegformerConfig(num_channels=num_channels,
                                             num_labels=num_labels,
                                             id2label=id2label,
                                             label2id=label2id,
                                             semantic_loss_ignore_index=0,
                                             depths=variant_config['depths'],
                                             hidden_sizes=variant_config['hidden_sizes'],
                                             decoder_hidden_size=variant_config['decoder_hidden_size'])

        self.model = SegformerForSemanticSegmentation(self.configuration)
        self.lr = lr

        self.metrics = {}
        for phase in ['train', 'val', 'test']:
            self.metrics[phase] = {
                'loss': MeanMetric().to(device),
                'iou': JaccardIndex(task='multiclass', num_classes=num_labels).to(device),
                'acc': Accuracy(task="multiclass", num_classes=num_labels).to(device),
                'f1': F1Score(task='multiclass', num_classes=num_labels).to(device),
                'precision': Precision(task='multiclass', num_classes=num_labels).to(device),
                'recall': Recall(task='multiclass', num_classes=num_labels).to(device)
            }

    def forward(self, x) -> Any:
        return self.model(x).logits

    def training_step(self, batch, *args):
        return self.__step(batch, phase='train')

    def validation_step(self, batch, *args):
        return self.__step(batch, phase='val')

    def test_step(self, batch, *args):
        return self.__step(batch, phase='test')

    def __step(self, batch, phase):
        x, y = batch['x'], batch['y']
        outputs = self.model(x, y)

        up_logits = F.interpolate(outputs.logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        y_pred = up_logits.argmax(dim=1)

        self.metrics[phase]['loss'].update(outputs.loss)
        self.log(f'{phase}/batch/loss', outputs.loss)

        for metric_name, metric in self.metrics[phase].items():
            if metric_name != 'loss':
                metric.update(y_pred, y)

        return outputs.loss

    def on_train_epoch_end(self):
        return self.__on_epoch_end('train')

    def on_validation_epoch_end(self):
        return self.__on_epoch_end('val')

    def on_test_epoch_end(self):
        return self.__on_epoch_end('test')

    def __on_epoch_end(self, phase):
        self.__log_metrics(phase)

    def __log_metrics(self, phase):
        for metric_name, metric in self.metrics[phase].items():
            self.log(f'{phase}/epoch/{metric_name}', metric.compute())
            metric.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
