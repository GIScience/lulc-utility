import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from onnxruntime import InferenceSession
from tifffile import imread

from app.api import app
from lulc.data.label import LabelDescriptor, resolve_osm_labels
from lulc.data.tx.array import Normalize, Stack, NanToNum, AdjustShape
from lulc.ops.imagery_store_operator import ImageryStore
from lulc.ops.osm_operator import OhsomeOps

TEST_JSON_start_end = {
    'area_coords': [
        7.3828125,
        47.5172006978394,
        7.55859375,
        47.63578359086485
    ],
    'start_date': '2023-05-01',
    'end_date': '2023-06-01'
}

TEST_JSON_end_only = {
    'area_coords': [
        7.3828125,
        47.5172006978394,
        7.55859375,
        47.63578359086485
    ],
    'end_date': '2023-06-01'
}

TEST_JSON_no_time = {
    'area_coords': [
        7.3828125,
        47.5172006978394,
        7.55859375,
        47.63578359086485
    ]
}

TEST_JSON_favour_osm = {
    'area_coords': [
        7.3828125,
        47.5172006978394,
        7.55859375,
        47.63578359086485
    ],
    'fusion_mode': 'favour_osm'
}

TEST_JSON_favour_model = {
    'area_coords': [
        7.3828125,
        47.5172006978394,
        7.55859375,
        47.63578359086485
    ],
    'fusion_mode': 'favour_model'
}

TEST_JSON_mean_mixin = {
    'area_coords': [
        7.3828125,
        47.5172006978394,
        7.55859375,
        47.63578359086485
    ],
    'fusion_mode': 'mean_mixin'
}

TODAY = datetime.now().date().isoformat()
WEEK_BEFORE = (datetime.now() - timedelta(days=7)).date().isoformat()

LABELS = [
    LabelDescriptor(name='unknown',
                    osm_filter='key=value',
                    color=(0, 0, 0),
                    description='description',
                    raster_value=0),
    LabelDescriptor(name='built-up',
                    osm_filter='key=value',
                    color=(255, 0, 0),
                    description='description',
                    raster_value=1),
    LabelDescriptor(name='forest',
                    osm_filter='key=value',
                    color=(77, 200, 0),
                    description='description',
                    raster_value=2),
    LabelDescriptor(name='water',
                    osm_filter='key=value',
                    color=(130, 200, 250),
                    description='description',
                    raster_value=3),
    LabelDescriptor(name='agriculture',
                    osm_filter='key=value',
                    color=(255, 255, 80),
                    description='description',
                    raster_value=4)
]


class TestImageryStore(ImageryStore):

    def __init__(self):
        self.last_start_date = None
        self.last_end_date = None

    def imagery(self,
                area_coords: Tuple,
                start_date: str,
                end_date: str,
                resolution: int = 10) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        self.last_start_date = start_date
        self.last_end_date = end_date

        return {
            's1.tif': np.random.uniform(0, 2, (512, 768, 2)).astype(np.float32),
            's2.tif': np.random.randint(0, 255, (512, 768, 6)).astype(np.float32),
            'dem.tif': np.random.uniform(0, 500, (512, 768, 1)).astype(np.float32),
        }, (512, 768)


class TestOhsomeOps(OhsomeOps):

    def __init__(self):
        super().__init__(Path('/tmp'))

    def labels(self, area_coords: Tuple[float, float, float, float],
               time: str,
               osm_lulc_mapping: Dict,
               target_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        output = {}
        for label in LABELS[1:]:
            output[label.name] = np.ones(target_size, dtype=np.int64)

        return output


@pytest.fixture
def mocked_client():
    client = TestClient(app)
    app.state.imagery_store = TestImageryStore()
    app.state.labels = LABELS
    app.state.osm = TestOhsomeOps()
    app.state.osm_lulc_mapping = resolve_osm_labels(app.state.labels)
    app.state.inference_session = InferenceSession(str(Path(__file__).parent / 'test.onnx'))

    def test_app_transformation_procedure(x):
        x = NanToNum(layers=['s1.tif', 's2.tif'], subset='imagery')(x)
        x = Stack(subset='imagery')(x)
        x = Normalize(subset='imagery', mean=np.random.random(9), std=np.random.random(9))(x)
        return AdjustShape(subset='imagery')(x)

    app.state.tx = test_app_transformation_procedure
    yield client


def test_health(mocked_client):
    response = mocked_client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


@pytest.mark.parametrize(
    'request_body, expected_start_date, expected_end_date',
    [(TEST_JSON_start_end, '2023-05-01', '2023-06-01'),
     (TEST_JSON_end_only, '2023-05-25', '2023-06-01'),
     (TEST_JSON_no_time, WEEK_BEFORE, TODAY),
     (TEST_JSON_favour_osm, WEEK_BEFORE, TODAY),
     (TEST_JSON_favour_model, WEEK_BEFORE, TODAY)
     ],
)
def test_segment_preview(mocked_client, request_body, expected_start_date, expected_end_date):
    response = mocked_client.post('/segment/preview', json=request_body)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert mocked_client.app.state.imagery_store.last_start_date == expected_start_date
    assert mocked_client.app.state.imagery_store.last_end_date == expected_end_date


def test_segment_image(mocked_client):
    response = mocked_client.post('/segment', json=TEST_JSON_start_end)
    response_data = imread(io.BytesIO(response.content))
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/geotiff'
    assert response_data.shape == (512, 768)


def test_segment_describe(mocked_client):
    response = mocked_client.get('/segment/describe')
    assert response.json() == {
        'unknown': {'name': 'unknown',
                    'osm_filter': 'key=value',
                    'color': [0, 0, 0],
                    'description': 'description',
                    'raster_value': 0, },
        'built-up': {'name': 'built-up',
                     'osm_filter': 'key=value',
                     'color': [255, 0, 0],
                     'description': 'description',
                     'raster_value': 1, },
        'forest': {'name': 'forest',
                   'osm_filter': 'key=value',
                   'color': [77, 200, 0],
                   'description': 'description',
                   'raster_value': 2, },
        'water': {'name': 'water',
                  'osm_filter': 'key=value',
                  'color': [130, 200, 250],
                  'description': 'description',
                  'raster_value': 3, },
        'agriculture': {'name': 'agriculture',
                        'osm_filter': 'key=value',
                        'color': [255, 255, 80],
                        'description': 'description',
                        'raster_value': 4, }}
