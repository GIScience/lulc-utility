import logging
from typing import List, Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from neptune.types import File
from torch.optim import Adam
from torchmetrics import MeanMetric, JaccardIndex, Accuracy, F1Score, Precision, Recall
from transformers import SegformerForSemanticSegmentation, \
    SegformerConfig

from lulc.model.metrics import ConfusionMatrix2D, PlotMetric

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

log = logging.getLogger(__name__)


class SegformerModule(pl.LightningModule):

    def __init__(self, num_channels: int,
                 labels: List[str],
                 variant: str,
                 lr: float,
                 device: torch.device,
                 class_weights: List[float],
                 color_codes: List[List[int]],
                 max_image_samples=5,
                 *args: Any,
                 **kwargs: Any):
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

        self.class_weights = torch.tensor(class_weights).to(device)
        self.color_codes = torch.tensor(color_codes).to(device)
        self.max_image_samples = max_image_samples

        self.model = SegformerForSemanticSegmentation(self.configuration)
        self.lr = lr

        self.images = {}
        self.metrics = {}
        for phase in ['train', 'val', 'test']:
            self.metrics[phase] = {
                'loss': MeanMetric(),
                'iou': JaccardIndex(task='multiclass', num_classes=num_labels),
                'acc': Accuracy(task="multiclass", num_classes=num_labels),
                'f1': F1Score(task='multiclass', num_classes=num_labels),
                'precision': Precision(task='multiclass', num_classes=num_labels),
                'recall': Recall(task='multiclass', num_classes=num_labels),
                'confusion_matrix': ConfusionMatrix2D(task='multiclass', num_classes=num_labels, normalize='true', labels=labels)
            }

    def on_fit_start(self) -> None:
        if torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled():
            log.warning('Deterministic transform has been set (warn only mode enabled)')
            torch.use_deterministic_algorithms(True, warn_only=True)

    def forward(self, x) -> Any:
        return self.model(x).logits

    def training_step(self, batch, *args):
        return self.step(batch, phase='train')

    def validation_step(self, batch, *args):
        return self.step(batch, phase='val')

    def test_step(self, batch, *args):
        return self.step(batch, phase='test')

    def step(self, batch, phase):
        x, y = batch['x'], batch['y']

        logits = self.model(x).logits
        upsampled_logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        loss = F.cross_entropy(upsampled_logits, y,
                               ignore_index=self.configuration.semantic_loss_ignore_index,
                               weight=self.class_weights)

        y_pred = upsampled_logits.argmax(dim=1)

        self.metrics[phase]['loss'].update(loss.cpu())
        self.log(f'{phase}/batch/loss', loss, prog_bar=True)
        self.register_sample_image(y_pred, phase)

        for metric_name, metric in self.metrics[phase].items():
            if metric_name != 'loss':
                metric.update(y_pred.cpu(), y.cpu())

        return loss

    def register_sample_image(self, y_pred: torch.Tensor, phase: str):
        if phase in self.images and self.images[phase].shape[0] < self.max_image_samples:
            self.images[phase] = torch.cat([self.images[phase], self.color_codes[y_pred]])[:self.max_image_samples]
        elif phase not in self.images:
            self.images[phase] = self.color_codes[y_pred]

    def on_train_epoch_end(self):
        return self.__on_epoch_end('train')

    def on_validation_epoch_end(self):
        return self.__on_epoch_end('val')

    def on_test_epoch_end(self):
        return self.__on_epoch_end('test')

    def __on_epoch_end(self, phase):
        self.__log_metrics(phase)
        self.__publish_images(phase)

    def __log_metrics(self, phase):
        for metric_name, metric in self.metrics[phase].items():
            if isinstance(metric, PlotMetric):
                image = File.as_image(metric.plot())
                self.logger.experiment[f'{phase}/epoch/{metric_name}'].append(image)
                plt.cla()
            else:
                self.log(f'{phase}/epoch/{metric_name}', metric.compute())
            metric.reset()

    def __publish_images(self, phase):
        images = self.images[phase].cpu().numpy()
        for idx in range(min(images.shape[0], self.max_image_samples)):
            image = File.as_image(images[idx] / 255)
            self.logger.experiment[f'{phase}/sample/image_{idx}'].append(image)

        del self.images[phase]

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
