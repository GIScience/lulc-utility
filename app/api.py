import io
import logging.config
import os
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import rasterio
import uvicorn
import yaml
from fastapi import APIRouter, FastAPI
from hydra import compose
from onnxruntime import InferenceSession
from PIL import Image
from pydantic import BaseModel, Field, confloat, model_validator
from rasterio.crs import CRS
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, Response

from app.predict import FusionMode, predict
from lulc.data.label import resolve_labels, resolve_osm_labels, resolve_corine_labels, LabelDescriptor
from lulc.data.tx.array import Normalize, Stack, NanToNum, AdjustShape
from lulc.model.ops.download import NeptuneModelDownload
from lulc.ops.imagery_store_operator import resolve_imagery_store
from lulc.ops.osm_operator import OhsomeOps

config_dir = os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute()))

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = f'{config_dir}/logging/app/logging.yaml'
log = logging.getLogger(__name__)

DATE_FORMAT = '%Y-%m-%d'


class HealthCheck(BaseModel):
    status: str = 'ok'


class ImageResponse(Response):
    media_type = 'image/png'


class GeoTiffResponse(FileResponse):
    media_type = 'image/geotiff'


@asynccontextmanager
async def configure_dependencies(app: FastAPI):
    """
    Initialize all required dependencies and attach them to the FastAPI state.
    Each underlying service utilizes configuration stored in `./conf/`.

    :param app: web application instance
    :return: context manager generator
    """
    log.info('Initialising...')

    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name='config')

    app.state.imagery_store = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))
    app.state.osm = OhsomeOps(cache_dir=Path(cfg.cache.dir))

    registry = NeptuneModelDownload(
        model_key=cfg.neptune.model.key,
        project=cfg.neptune.project,
        api_key=cfg.neptune.api_token,
        cache_dir=Path(cfg.cache.dir),
    )

    onnx_model, label_descriptor_version = registry.download_model_version(cfg.serve.model_version)

    app.state.labels = resolve_labels(Path(cfg.data.dir), label_descriptor_version)
    app.state.osm_lulc_mapping = resolve_osm_labels(app.state.labels)
    app.state.corine_labels = resolve_corine_labels(Path(cfg.data.dir))
    app.state.inference_session = InferenceSession(str(onnx_model))

    def tx(x):
        x = NanToNum(layers=['s1.tif', 's2.tif'], subset='imagery')(x)
        x = Stack(subset='imagery')(x)
        x = Normalize(subset='imagery', mean=cfg.data.normalize.mean, std=cfg.data.normalize.std)(x)
        return AdjustShape(subset='imagery')(x)

    app.state.tx = tx

    log.info('Initialisation completed')

    yield


class LulcWorkUnit(BaseModel):
    """LULC area of interest."""

    area_coords: Tuple[
        confloat(ge=-180, le=180), confloat(ge=-90, le=90), confloat(ge=-180, le=180), confloat(ge=-90, le=90)
    ] = Field(
        title='Area Coordinates',
        description='Bounding box coordinates in WGS 84 (west, south, east, north)',
        examples=[
            [
                12.304687500000002,
                48.2246726495652,
                12.480468750000002,
                48.3416461723746,
            ]
        ],
    )
    start_date: Optional[date] = Field(
        title='Start Date',
        description='Lower bound (inclusive) of remote sensing imagery acquisition date (UTC). '
        'The model uses an image stack of multiple acquisition times for predictions. '
        'Larger time intervals will improve the prediction accuracy'
        'If not set it will be automatically set to the week before `end_date`',
        examples=['2023-05-01'],
        default=None,
    )
    end_date: date = Field(
        title='End Date',
        description='Upper bound (inclusive) of remote sensing imagery acquisition date (UTC).'
        "Defaults to today's date"
        'In case `fusion_mode` has been declared to value different than `only_model`'
        'the `end_date` will also be used to acquire OSM data',
        examples=['2023-06-01'],
        default=datetime.now().date(),
    )
    threshold: confloat(ge=0.0, le=1.0) = Field(
        title='Threshold',
        description='Not exceeding this value by the class prediction score results in the recognition of the result '
        'as "unknown"',
        default=0,
        examples=[0.75],
    )
    fusion_mode: FusionMode = Field(
        title='Fusion Mode',
        description='Enables merging model results with OSM data: '
        '`only_model` - no fusion with OSM will take place, '
        '`only_osm` - displays OSM output only, '
        '`favour_model` - OSM will be used to fill in regions considered as '
        '"unknown" for the model, '
        '`favour_osm` - model results will be used to fill in empty OSM data, '
        '`mean_mixin` - model and OSM will simultaneously contribute to '
        'overall classification',
        default=FusionMode.ONLY_MODEL,
        examples=[FusionMode.ONLY_MODEL],
    )

    @model_validator(mode='after')
    def minus_week(self) -> 'LulcWorkUnit':
        if not self.start_date:
            self.start_date = self.end_date - timedelta(days=7)
        return self


health = APIRouter(prefix='/health')
segment = APIRouter(prefix='/segment')


@health.get('', status_code=200, description='Verify whether the application API is operational')
def is_ok() -> HealthCheck:
    return HealthCheck()


@segment.post(
    '/preview',
    description='Run the semantic segmentation algorithm and return a preview of the result '
    '(1/4 target low-resolution, medium-compression image)',
    response_class=ImageResponse,
)
def segment_preview(body: LulcWorkUnit, request: Request) -> ImageResponse:
    log.info(f'Creating preview for {body}')

    labels, _ = predict(
        request.app.state.imagery_store,
        request.app.state.osm,
        request.app.state.inference_session,
        request.app.state.tx,
        request.app.state.osm_lulc_mapping,
        body.threshold,
        body.area_coords,
        body.start_date.isoformat(),
        body.end_date.isoformat(),
        body.fusion_mode,
    )

    if body.fusion_mode == FusionMode.ONLY_CORINE:
        labels = np.array([d.color for d in request.app.state.corine_labels], dtype=np.uint8)[labels]

    else:
        labels = np.array([d.color for d in request.app.state.labels], dtype=np.uint8)[labels]

    labels = Image.fromarray(labels)

    buf = io.BytesIO()
    labels.save(buf, format='PNG')
    buf.seek(0)

    log.info(f'Finished creating preview for {body}')

    return ImageResponse(buf.getvalue(), media_type='image/png')


@segment.post(
    '/',
    description='Run the semantic segmentation algorithm and return a georeferenced raster (GeoTIFF)',
    response_class=GeoTiffResponse,
)
def segment_compute(body: LulcWorkUnit, request: Request) -> GeoTiffResponse:
    log.info(f'Creating segementation for {body}')

    labels, (h, w) = predict(
        request.app.state.imagery_store,
        request.app.state.osm,
        request.app.state.inference_session,
        request.app.state.tx,
        request.app.state.osm_lulc_mapping,
        body.threshold,
        body.area_coords,
        body.start_date.isoformat(),
        body.end_date.isoformat(),
        body.fusion_mode,
    )

    transform = rasterio.transform.from_bounds(*body.area_coords, width=w, height=h)
    file_path = Path(f'/tmp/{uuid.uuid4()}.tiff')

    def unlink():
        file_path.unlink()

    with rasterio.open(
        file_path,
        mode='w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype=labels.dtype,
        crs=CRS.from_string('EPSG:4326'),
        nodata=None,
        transform=transform,
    ) as dst:
        dst.write(labels, 1)
        if body.fusion_mode == FusionMode.ONLY_CORINE:
            dst.write_colormap(1, dict(enumerate([d.color for d in request.app.state.corine_labels])))
        else:
            dst.write_colormap(1, dict(enumerate([d.color for d in request.app.state.labels])))

    log.info(f'Finished creating segmentation for {body}')

    return GeoTiffResponse(
        file_path,
        media_type='image/geotiff',
        filename='segmentation.tiff',
        background=BackgroundTask(unlink),
    )


@segment.get('/describe', description='Return semantic segmentation labels dictionary')
def segment_describe(request: Request) -> Dict[str, LabelDescriptor]:
    return {label.name: label for label in request.app.state.labels}


app = FastAPI(lifespan=configure_dependencies)
app.include_router(segment)
app.include_router(health)

if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))
    log.info('Starting LULC Utility')
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=int(os.getenv('LULC_UTILITY_API_PORT', 8000)),
        root_path=os.getenv('ROOT_PATH', '/'),
        log_config=log_config,
        log_level=log_level.lower(),
    )
