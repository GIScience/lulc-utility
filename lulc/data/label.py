from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from PIL import ImageColor

BACKGROUND_COLOR = [0, 0, 0]
BACKGROUND_LABEL = 'unknown'


@dataclass
class LabelsDescriptor:
    color_codes: List[List[int]]
    names: List[str]


def resolve_labels(data_dir: Path, label_descriptor_version: str) -> LabelsDescriptor:
    def aux(color: str):
        r, g, b = ImageColor.getcolor(color, 'RGB')
        return pd.Series([r, g, b])

    labels_descriptor = pd.read_csv(str(data_dir / 'label' / f'label_{label_descriptor_version}.csv'),
                                    index_col='label')['color_code']
    label_color_codes = labels_descriptor.apply(aux).values
    return LabelsDescriptor(
        [BACKGROUND_COLOR] + label_color_codes.tolist(),
        [BACKGROUND_LABEL] + labels_descriptor.index.values.tolist()
    )
