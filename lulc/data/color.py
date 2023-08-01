from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageColor

BACKGROUND_COLOR = [0, 0, 0]


def resolve_color_codes(data_dir: Path, label_descriptor_version: str):
    def aux(color: str):
        r, g, b = ImageColor.getcolor(color, 'RGB')
        return pd.Series([r, g, b])

    label_color_codes = pd.read_csv(str(data_dir / 'label' / f'label_{label_descriptor_version}.csv'),
                                    index_col='label')['color_code']
    label_color_codes = label_color_codes.apply(aux).values
    return np.append([BACKGROUND_COLOR], label_color_codes, axis=0).astype(np.uint8)
