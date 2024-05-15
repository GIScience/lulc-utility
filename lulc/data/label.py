from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yaml
from PIL import ImageColor
from pydantic import BaseModel, Field


class LabelDescriptor(BaseModel):
    """Segmentation label definition."""

    name: str = Field(title='Name', description='Name of the segmentation label.', examples=['Forest'])
    description: Optional[str] = Field(
        title='Description',
        description='A concise label description or caption.',
        examples=['Areas with a tree cover of more than 80% and a continuous area of more than 25ha'],
        default=None,
    )
    osm_filter: Optional[str] = Field(
        title='OSM Filter',
        description='The OSM filter statement that will extract all elements that fit '
        'the description of this label.',
        examples=['landuse=forest or natural=wood'],
        default=None,
    )
    raster_value: int = Field(
        title='Raster Value', description='The numeric value in the raster that represents this label.', examples=[1]
    )
    color: Tuple[int, int, int] = Field(
        title='Color Hex-Code', description='The RGB-color values of the label', examples=[(255, 0, 0)]
    )


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self))


BACKGROUND_DESCRIPTOR = LabelDescriptor(
    name='unknown',
    raster_value=0,
    color=(0, 0, 0),
    description='areas which class cannot be predicted with a required certainty score',
)


def resolve_labels(data_dir: Path, label_descriptor_version: str) -> List[LabelDescriptor]:
    def aux(color: str) -> Tuple[int, int, int]:
        r, g, b = ImageColor.getcolor(color, 'RGB')
        return r, g, b

    with open(str(data_dir / 'label' / f'label_{label_descriptor_version}.yaml'), 'r') as file:
        labels_descriptor = pd.DataFrame(yaml.safe_load(file))
        labels_descriptor['raster_value'] = pd.RangeIndex(1, len(labels_descriptor) + 1)
        labels_descriptor['color_code'] = labels_descriptor['color_code'].apply(aux).values
        labels_descriptor = labels_descriptor.rename(columns={'color_code': 'color'})
        return [BACKGROUND_DESCRIPTOR] + [LabelDescriptor(**row) for _, row in labels_descriptor.iterrows()]


def resolve_osm_labels(labels: List[LabelDescriptor]) -> HashableDict:
    labels_dict = dict([(d.name, d.osm_filter) for d in labels if d.osm_filter is not None])
    return HashableDict(labels_dict)


def resolve_corine_labels(data_dir: Path) -> List[LabelDescriptor]:
    def aux(color: str) -> Tuple[int, int, int]:
        r, g, b = ImageColor.getcolor(color, 'RGB')
        return r, g, b

    with open(str(data_dir / 'label' / 'corine.yaml'), 'r') as file:
        labels_descriptor = pd.DataFrame(yaml.safe_load(file))
        labels_descriptor['raster_value'] = pd.RangeIndex(1, len(labels_descriptor) + 1)
        labels_descriptor['color'] = labels_descriptor['color'].apply(aux).values
        return [BACKGROUND_DESCRIPTOR] + [LabelDescriptor(**row) for _, row in labels_descriptor.iterrows()]
