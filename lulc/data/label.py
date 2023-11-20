from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from PIL import ImageColor


@dataclass
class LabelDescriptor:
    name: str
    filter: Optional[str]
    color_code: List[int]
    description: Optional[str]


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self))


BACKGROUND_DESCRIPTOR = LabelDescriptor(
    'unknown',
    None,
    [0, 0, 0],
    'areas which class cannot be predicted with a required certainty score'
)


def resolve_labels(data_dir: Path, label_descriptor_version: str) -> List[LabelDescriptor]:
    def aux(color: str):
        r, g, b = ImageColor.getcolor(color, 'RGB')
        return [r, g, b]

    with open(str(data_dir / 'label' / f'label_{label_descriptor_version}.yaml'), 'r') as file:
        labels_descriptor = pd.DataFrame(yaml.safe_load(file))
        labels_descriptor['color_code'] = labels_descriptor['color_code'].apply(aux).values
        return [BACKGROUND_DESCRIPTOR] + [LabelDescriptor(**row) for _, row in labels_descriptor.iterrows()]


def resolve_osm_labels(labels: List[LabelDescriptor]) -> HashableDict:
    labels_dict = dict([(d.name, d.filter) for d in labels if d.filter is not None])
    return HashableDict(labels_dict)
