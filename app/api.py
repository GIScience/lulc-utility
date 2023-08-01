import io
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import uvicorn
from PIL import Image
from PIL.Image import Resampling
from fastapi import FastAPI, APIRouter, HTTPException
from hydra import compose
from onnxruntime import InferenceSession
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response, RedirectResponse
from torchvision import transforms

from lulc.data.color import resolve_color_codes
from lulc.data.tx import MinMaxScaling, MaxScaling, Stack, NanToNum, ExtendShape
from lulc.model.registry import NeptuneModelRegistry
from lulc.ops.exception import OperatorValidationException, OperatorInteractionException
from lulc.ops.sentinelhub_operator import SentinelHubOperator, ImageryStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def configure_dependencies(app: FastAPI):
    hydra.initialize_config_dir(config_dir=os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute())), version_base=None)
    cfg = compose(config_name='config')

    app.state.imagery_store = SentinelHubOperator(api_id=cfg.sentinel_hub.api.id,
                                                  api_secret=cfg.sentinel_hub.api.secret,
                                                  evalscript_dir=Path(cfg.data.dir) / 'imagery',
                                                  evalscript_name=f'imagery_{cfg.data.descriptor.imagery}',
                                                  cache_dir=Path(cfg.cache.dir) / 'sentinelhub')

    app.state.color_codes = resolve_color_codes(Path(cfg.data.dir), cfg.data.descriptor.labels)

    registry = NeptuneModelRegistry(model_key=cfg.neptune.model.key,
                                    project=cfg.neptune.project,
                                    api_key=cfg.neptune.api_token,
                                    cache_dir=Path(cfg.cache.dir))
    onnx_model = registry.download_model_version(cfg.serve.model_version)
    app.state.inference_session = InferenceSession(str(onnx_model))
    yield


@lru_cache(maxsize=32)
def __predict(imagery_store: ImageryStore, inference_session: InferenceSession, area_coords: Tuple[float, float, float, float], start_date: str, end_date: str):
    try:
        imagery, imagery_size = imagery_store.imagery(area_coords, start_date, end_date)
        tx = transforms.Compose([
            NanToNum(layers=['s1.tif', 's2.tif'], subset='imagery'),
            MinMaxScaling(layers=['s1.tif', 'dem.tif'], subset='imagery'),
            MaxScaling(layers=['s2.tif'], subset='imagery'),
            Stack(subset='imagery'),
            ExtendShape(subset='imagery', ch_first=True),
        ])
        imagery = tx({'imagery': imagery})
        labels = inference_session.run(output_names=None, input_feed=imagery)[0][0]

        return np.argmax(labels, axis=0, keepdims=False).astype(np.uint8), imagery_size
    except OperatorValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OperatorInteractionException as e:
        raise HTTPException(status_code=500, detail=str(e))


class Body(BaseModel):
    area_coords: Tuple[float, float, float, float] = Field(examples=[[7.3828125, 47.5172006978394, 7.55859375, 47.63578359086485]])
    start_date: str = Field(examples=['2023-05-01'])
    end_date: str = Field(examples=['2023-06-01'])


health = APIRouter(prefix='/health')
segment = APIRouter(prefix='/segment')


@health.get('', status_code=200)
def is_ok():
    return {'status': 'ok'}


@segment.post('/preview')
def segment_preview(body: Body, request: Request):
    labels, _ = __predict(request.app.state.imagery_store, request.app.state.inference_session, body.area_coords, body.start_date, body.end_date)
    labels = request.app.state.color_codes[labels]
    labels = Image.fromarray(labels)

    buf = io.BytesIO()
    labels.save(buf, format='PNG')
    buf.seek(0)

    return Response(buf.getvalue(), media_type='image/png')


@segment.post('/')
def segment_compute(body: Body, request: Request):
    labels, (h, w) = __predict(request.app.state.imagery_store, request.app.state.inference_session, body.area_coords, body.start_date, body.end_date)
    labels = Image.fromarray(labels, mode='L').resize((w, h), Resampling.BILINEAR)

    buf = io.BytesIO()
    labels.save(buf, format='TIFF')
    buf.seek(0)

    return Response(buf.getvalue(), media_type='image/tiff')


version = APIRouter(prefix='/v1')
version.include_router(segment)

app = FastAPI(lifespan=configure_dependencies)
app.include_router(version)
app.include_router(health)


@app.get('/', include_in_schema=False)
def docs_redirect():  # dead:disable
    """Redirect to the documentation of the API."""
    return RedirectResponse(url='/docs')


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('LULC_UTILITY_API_PORT', 8000)))
