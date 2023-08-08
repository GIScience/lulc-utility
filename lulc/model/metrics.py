from abc import abstractmethod
from typing import Optional, Literal, Any, List

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric, ConfusionMatrix


class PlotMetric(Metric):

    @abstractmethod
    def plot(self):
        pass


class ConfusionMatrix2D(PlotMetric):

    def __init__(self,
                 labels: List[str],
                 task: Literal["binary", "multiclass", "multilabel"],
                 threshold: float = 0.5,
                 num_classes: Optional[int] = None,
                 num_labels: Optional[int] = None,
                 normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
                 ignore_index: Optional[int] = None,
                 validate_args: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cm = ConfusionMatrix(task, threshold, num_classes, num_labels, normalize, ignore_index, validate_args)
        self.labels = labels

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.cm.update(preds.flatten(start_dim=1), target.flatten(start_dim=1))

    def compute(self) -> Tensor:
        return self.cm.compute()

    def plot(self) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax = sns.heatmap(data=self.compute().cpu().numpy(),
                         xticklabels=self.labels,
                         yticklabels=self.labels,
                         cbar=False,
                         annot=True,
                         square=True,
                         cmap='RdYlGn',
                         ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        return fig

    def reset(self):
        self.cm.reset()
        super().reset()
