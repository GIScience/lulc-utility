from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from onnxruntime import InferenceSession

from app.process import FusionMode, analyse
from lulc.data.label import HashableDict, LabelDescriptor
from lulc.data.tx.array import AdjustShape, NanToNum, Stack
from lulc.ops.imagery_store_operator import ImageryStore
from lulc.ops.osm_operator import OhsomeOps

OSM_LABEL_MAPPING = HashableDict(
    {
        'unknown': LabelDescriptor(name='unknown', raster_value=0, color=(0, 0, 0)),
        'osm_1': LabelDescriptor(name='osm_1', osm_filter='filter_1', raster_value=0, color=(0, 0, 0)),
        'osm_2': LabelDescriptor(name='osm_2', osm_filter='filter_2', raster_value=0, color=(0, 0, 0)),
    }
)

CORINE_LABEL_MAPPING = HashableDict(
    {
        'unknown': LabelDescriptor(name='unknown', osm_filter='filter_1', raster_value=0, color=(0, 0, 0)),
        'corine_1': LabelDescriptor(name='corine_1', osm_ref='osm_1', raster_value=0, color=(0, 0, 0)),
        'corine_2': LabelDescriptor(name='corine_2', osm_ref='osm_1', raster_value=0, color=(0, 0, 0)),
        'corine_3': LabelDescriptor(name='corine_3', osm_ref='osm_2', raster_value=0, color=(0, 0, 0)),
    }
)


@pytest.fixture
def imagery_store():
    mock = Mock()
    mock.imagery.return_value = (
        {
            's1.tif': np.random.rand(256, 256, 2),
            's2.tif': np.random.rand(256, 256, 6),
            'dem.tif': np.random.rand(256, 256, 2),
        },
        (256, 256),
    )

    mock.labels.return_value = np.random.randint(0, 3, (128, 128), dtype=np.uint8), (128, 128)

    yield mock


@pytest.fixture
def osm_ops():
    mock = Mock()
    mock.labels.return_value = {
        'built_up': np.random.randint(0, 1, (64, 48)).astype(np.bool_),
        'forest': np.random.randint(0, 1, (64, 48)).astype(np.bool_),
    }
    yield mock


@pytest.fixture
def inference_session():
    mock = Mock()
    mock.run.return_value = [np.random.rand(1, 3, 64, 48)]
    yield mock


@pytest.mark.parametrize(
    'fusion_mode, expected_imagery_size, expected_labels_shape',
    [
        (FusionMode.ONLY_MODEL, (256, 256), (64, 48)),
        (FusionMode.ONLY_OSM, (256, 256), (64, 48)),
        (FusionMode.ONLY_CORINE, (128, 128), (128, 128)),
        (FusionMode.FAVOUR_MODEL, (256, 256), (64, 48)),
        (FusionMode.FAVOUR_OSM, (256, 256), (64, 48)),
        (FusionMode.MEAN_MIXIN, (256, 256), (64, 48)),
        (FusionMode.HARMONIZED_CORINE, (128, 128), (128, 128)),
    ],
)
def test_analyse_when_fusion_various_fusion_modes(
    fusion_mode: FusionMode,
    expected_imagery_size: Tuple[int, int],
    expected_labels_shape: Tuple[int, int],
    imagery_store: ImageryStore,
    osm_ops: OhsomeOps,
    inference_session: InferenceSession,
):
    def test_tx(x):
        x = NanToNum(layers=['s1.tif', 's2.tif'], subset='imagery')(x)
        x = Stack(subset='imagery')(x)
        return AdjustShape(subset='imagery')(x)

    labels, imagery_size = analyse(
        imagery_store,
        osm_ops,
        inference_session,
        test_tx,
        OSM_LABEL_MAPPING,
        CORINE_LABEL_MAPPING,
        threshold=0.75,
        area_coords=(0, 0, 0, 0),
        start_date='2024-04-01',
        end_date='2024-04-01',
        fusion_mode=fusion_mode,
    )
    assert labels.shape == expected_labels_shape
    assert imagery_size == expected_imagery_size
    assert labels.dtype == np.uint8
