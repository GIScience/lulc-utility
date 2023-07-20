from typing import List, Any

import lightning.pytorch as pl
from torch.optim import Adam
from transformers import SegformerForSemanticSegmentation, \
    SegformerConfig


class SegFormerModule(pl.LightningModule):

    def __init__(self, num_channels: int, labels: List[str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        num_labels = len(labels)
        label2id = {k: v for v, k in enumerate(labels)}
        id2label = {v: k for k, v in label2id.items()}

        configuration = SegformerConfig(num_channels=num_channels, num_labels=num_labels, id2label=id2label,
                                        label2id=label2id, semantic_loss_ignore_index=0)
        self.model = SegformerForSemanticSegmentation(configuration)

    def training_step(self, batch):
        x, y = batch['x'], batch['y']
        loss = self.model(x, y).loss
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
