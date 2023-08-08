import io
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from onnxruntime import InferenceSession
from tifffile import imread

from app.api import app
from lulc.data.label import LabelsDescriptor
from lulc.data.tx.array import Normalize, Stack, NanToNum, AdjustShape
from lulc.ops.sentinelhub_operator import ImageryStore


class TestImageryStore(ImageryStore):

    def imagery(self, area_coords: Tuple, start_date: str, end_date: str,
                resolution: int = 10) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        return {
            's1.tif': np.random.uniform(0, 2, (512, 768, 2)).astype(np.float32),
            's2.tif': np.random.randint(0, 255, (512, 768, 6)).astype(np.float32),
            'dem.tif': np.random.uniform(0, 500, (512, 768, 1)).astype(np.float32),
        }, (512, 768)


TEST_JSON = {
    "area_coords": [
        7.3828125,
        47.5172006978394,
        7.55859375,
        47.63578359086485
    ],
    "start_date": "2023-05-01",
    "end_date": "2023-06-01"
}


@pytest.fixture
def mocked_client():
    client = TestClient(app)
    app.state.imagery_store = TestImageryStore()
    app.state.labels = LabelsDescriptor(
        [[0, 0, 0], [255, 0, 0], [77, 200, 0], [130, 200, 250], [255, 255, 80]],
        ["unknown", "built-up", "forest", "water", "agriculture"]
    )
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


def test_segment_preview(mocked_client):
    response = mocked_client.post('/segment/preview', json=TEST_JSON)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'


def test_segment_image(mocked_client):
    response = mocked_client.post('/segment', json=TEST_JSON)
    response_data = imread(io.BytesIO(response.content))
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/geotiff'
    assert response_data.shape == (512, 768)


def test_segment_describe(mocked_client):
    response = mocked_client.get('/segment/describe')
    assert response.json() == {'color_codes': [[0, 0, 0],
                                               [255, 0, 0],
                                               [77, 200, 0],
                                               [130, 200, 250],
                                               [255, 255, 80]],
                               'names': ['unknown',
                                         'built-up',
                                         'forest',
                                         'water',
                                         'agriculture']}
