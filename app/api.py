import io
import logging
import os
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Callable

import hydra
import numpy as np
import rasterio
import uvicorn
from PIL import Image
from fastapi import FastAPI, APIRouter, HTTPException
from hydra import compose
from onnxruntime import InferenceSession
from pydantic import BaseModel, Field
from rasterio.crs import CRS
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, FileResponse

from lulc.data.label import resolve_labels
from lulc.data.tx.array import Normalize, Stack, NanToNum, AdjustShape
from lulc.model.ops.download import NeptuneModelDownload
from lulc.ops.exception import OperatorValidationException, OperatorInteractionException
from lulc.ops.imagery_store_operator import ImageryStore, resolve_imagery_store

log = logging.getLogger(__name__)

config_dir = os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute()))


@asynccontextmanager
async def configure_dependencies(app: FastAPI):
    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name='config')

    app.state.imagery_store = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))

    app.state.labels = resolve_labels(Path(cfg.data.dir), cfg.data.descriptor.labels)

    registry = NeptuneModelDownload(model_key=cfg.neptune.model.key,
                                    project=cfg.neptune.project,
                                    api_key=cfg.neptune.api_token,
                                    cache_dir=Path(cfg.cache.dir))
    onnx_model = registry.download_model_version(cfg.serve.model_version)
    app.state.inference_session = InferenceSession(str(onnx_model))

    def tx(x):
        x = NanToNum(layers=['s1.tif', 's2.tif'], subset='imagery')(x)
        x = Stack(subset='imagery')(x)
        x = Normalize(subset='imagery', mean=cfg.data.normalize.mean, std=cfg.data.normalize.std)(x)
        return AdjustShape(subset='imagery')(x)

    app.state.tx = tx
    yield


@lru_cache(maxsize=32)
def __predict(imagery_store: ImageryStore, inference_session: InferenceSession, tx: Callable,
              area_coords: Tuple[float, float, float, float], start_date: str, end_date: str):
    try:
        imagery, imagery_size = imagery_store.imagery(area_coords, start_date, end_date)
        imagery = tx({'imagery': imagery})
        logits = inference_session.run(output_names=None, input_feed=imagery)[0][0]

        return np.argmax(logits, axis=0, keepdims=False).astype(np.uint8), imagery_size
    except OperatorValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OperatorInteractionException as e:
        raise HTTPException(status_code=500, detail=str(e))


class Body(BaseModel):
    area_coords: Tuple[float, float, float, float] = Field(examples=[[12.304687500000002,
                                                                      48.2246726495652,
                                                                      12.480468750000002,
                                                                      48.3416461723746]])
    start_date: str = Field(examples=['2023-05-01'])
    end_date: str = Field(examples=['2023-06-01'])


health = APIRouter(prefix='/health')
segment = APIRouter(prefix='/segment')


@health.get('', status_code=200)
def is_ok():
    return {'status': 'ok'}


@segment.post('/preview')
def segment_preview(body: Body, request: Request):
    labels, _ = __predict(request.app.state.imagery_store,
                          request.app.state.inference_session,
                          request.app.state.tx,
                          body.area_coords,
                          body.start_date,
                          body.end_date)
    labels = np.array(request.app.state.labels.color_codes, dtype=np.uint8)[labels]
    labels = Image.fromarray(labels)

    buf = io.BytesIO()
    labels.save(buf, format='PNG')
    buf.seek(0)

    return Response(buf.getvalue(), media_type='image/png')


@segment.post('/')
def segment_compute(body: Body, request: Request):
    labels, (h, w) = __predict(request.app.state.imagery_store,
                               request.app.state.inference_session,
                               request.app.state.tx,
                               body.area_coords,
                               body.start_date,
                               body.end_date)

    transform = rasterio.transform.from_bounds(*body.area_coords, width=w, height=h)
    file_path = Path(f'/tmp/{uuid.uuid4()}.tiff')

    def unlink():
        file_path.unlink()

    with rasterio.open(file_path,
                       mode='w',
                       driver='GTiff',
                       height=h,
                       width=w,
                       count=1,
                       dtype=labels.dtype,
                       crs=CRS.from_string('EPSG:4326'),
                       nodata=None,
                       transform=transform) as dst:
        dst.write(labels, 1)
        dst.write_colormap(1, dict(enumerate(request.app.state.labels.color_codes)))

    return FileResponse(file_path,
                        media_type='image/geotiff',
                        filename='segmentation.tiff',
                        background=BackgroundTask(unlink))


@segment.get('/describe')
def segment_describe(request: Request):
    return request.app.state.labels


app = FastAPI(lifespan=configure_dependencies)
app.include_router(segment)
app.include_router(health)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('LULC_UTILITY_API_PORT', 8000)), log_config=f'{config_dir}/logging/app/logging.yaml')
