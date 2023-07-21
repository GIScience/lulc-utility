from typing import List, Any

import lightning.pytorch as pl
import torch
from torch.optim import Adam
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


class SegFormerModule(pl.LightningModule):

    def __init__(self, num_channels: int, labels: List[str], variant: str, lr: float,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        num_labels = len(labels)
        label2id = {k: v for v, k in enumerate(labels)}
        id2label = {v: k for k, v in label2id.items()}

        variant_config = MODEL_VARIANTS[variant]
        configuration = SegformerConfig(num_channels=num_channels,
                                        num_labels=num_labels,
                                        id2label=id2label,
                                        label2id=label2id,
                                        semantic_loss_ignore_index=0,
                                        depths=variant_config['depths'],
                                        hidden_sizes=variant_config['hidden_sizes'],
                                        decoder_hidden_size=variant_config['decoder_hidden_size'])

        self.model = SegformerForSemanticSegmentation(configuration)
        self.lr = lr

        self.training_step_outputs = []

    def training_step(self, batch):
        x, y = batch['x'], batch['y']
        loss = self.model(x, y).loss
        self.log('metrics/batch/loss', loss, prog_bar=False)
        self.training_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.training_step_outputs).mean()
        self.log('metrics/loss', loss)
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
