import io
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from fastapi.testclient import TestClient
from onnxruntime import InferenceSession
from tifffile import imread

from app.api import app
from lulc.ops.sentinelhub_operator import ImageryStore


class TestImageryStore(ImageryStore):

    def imagery(self, area_coords: Tuple, start_date: str, end_date: str,
                resolution: int = 10) -> tuple[Dict[str, np.ndarray], tuple[int, int]]:
        return {
            's1.tif': np.random.uniform(0, 2, (512, 768, 2)),
            's2.tif': np.random.randint(0, 255, (512, 768, 6)),
            'dem.tif': np.random.uniform(0, 500, (512, 768, 1)),
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

client = TestClient(app)
app.state.imagery_store = TestImageryStore()
app.state.color_codes = np.random.randint(0, 255, (5, 3)).astype(np.uint8)
app.state.inference_session = InferenceSession(str(Path(__file__).parent / 'test.onnx'))


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_segment_preview():
    response = client.post('/v1/segment/preview', json=TEST_JSON)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'


def test_segment_image():
    response = client.post('/v1/segment', json=TEST_JSON)
    response_data = imread(io.BytesIO(response.content))
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/tiff'
    assert response_data.shape == (512, 768)
