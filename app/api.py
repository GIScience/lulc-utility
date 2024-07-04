import io
import logging.config
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import hydra
import numpy as np
import rasterio
import uvicorn
import yaml
from PIL import Image
from PIL.Image import Resampling
from fastapi import APIRouter, FastAPI
from hydra import compose
from onnxruntime import InferenceSession
from pydantic import BaseModel, Field, confloat, model_validator
from rasterio.crs import CRS
from rasterio.mask import mask
from sentinelhub import BBox, CRS as SCRS, to_utm_bbox
from shapely import Polygon
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, Response

from app.process import FusionMode, analyse
from lulc.data.label import resolve_corine_labels, LabelDescriptor, resolve_osm_labels, HashableDict
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


class LabelResponse(BaseModel):
    osm: Dict[str, LabelDescriptor] = Field(
        title='OSM Labels',
        description='Labels of classes present in the osm derived data.',
        examples=[
            {
                'corine': LabelDescriptor(
                    name='unknown',
                    osm_filter=None,
                    color=(0, 0, 0),
                    description='Class Unknown',
                    raster_value=0,
                )
            }
        ],
    )
    corine: Dict[str, LabelDescriptor] = Field(
        title='CORINE Labels',
        description='Labels of classes present in the corine derived data.',
        examples=[
            {
                'corine': LabelDescriptor(
                    name='unknown',
                    osm_filter=None,
                    color=(0, 0, 0),
                    description='Class Unknown',
                    raster_value=0,
                )
            }
        ],
    )


@dataclass
class AnalysisResult:
    labels: np.ndarray
    height: int
    width: int
    area_coords: Tuple[float, float, float, float]
    clip_geometry: Polygon


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

    app.state.osm_labels = resolve_osm_labels(Path(cfg.data.dir), label_descriptor_version)
    app.state.osm_cmap = dict(enumerate([d.color for d in app.state.osm_labels]))

    app.state.corine_labels = resolve_corine_labels(Path(cfg.data.dir), label_descriptor_version)
    app.state.corine_cmap = dict(enumerate([d.color for d in app.state.corine_labels]))

    app.state.inference_session = InferenceSession(str(onnx_model))

    def tx(x):
        x = NanToNum(layers=['s1.tif', 's2.tif'], subset='imagery')(x)
        x = Stack(subset='imagery')(x)
        x = Normalize(subset='imagery', mean=cfg.data.normalize.mean, std=cfg.data.normalize.std)(x)
        return AdjustShape(subset='imagery')(x)

    app.state.tx = tx
    app.state.edge_smoothing_buffer = cfg.serve.edge_smoothing_buffer

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
        'overall classification'
        '`only_corine` - return raw CLC data'
        '`harmonized_corine` - return CLC data reclassified to match model output',
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
async def is_ok() -> HealthCheck:
    return HealthCheck()


@segment.post(
    '/preview',
    description='Run the semantic segmentation algorithm and return a preview of the result '
    '(low-resolution, medium-compression image)',
    response_class=ImageResponse,
)
async def segment_preview(body: LulcWorkUnit, request: Request) -> ImageResponse:
    log.info(f'Creating preview for {body}')
    result = __analyse(body, request)

    labels = (
        request.app.state.corine_labels if body.fusion_mode == FusionMode.ONLY_CORINE else request.app.state.osm_labels
    )

    labels = np.array([d.color for d in labels], dtype=np.uint8)[result.labels]
    labels = Image.fromarray(labels)
    labels = labels.resize(
        (512, 512),
        resample=Resampling.NEAREST,
    )

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
async def segment_compute(body: LulcWorkUnit, request: Request) -> GeoTiffResponse:
    log.info(f'Creating segmentation for {body}')
    result = __analyse(body, request)

    file_path = Path(f'/tmp/{uuid.uuid4()}.tiff')

    def unlink():
        file_path.unlink()

    with rasterio.open(
        file_path,
        mode='w+',
        driver='GTiff',
        height=result.height,
        width=result.width,
        count=1,
        dtype=result.labels.dtype,
        crs=CRS.from_string('EPSG:4326'),
        nodata=None,
        transform=rasterio.transform.from_bounds(*result.area_coords, width=result.width, height=result.height),
    ) as dst:
        dst.write(result.labels, 1)

        masked_image, out_transform = mask(dst, shapes=[result.clip_geometry], crop=True)
        out_profile = dst.profile.copy()

    out_profile.update(
        {
            'driver': 'GTiff',
            'height': masked_image.shape[1],
            'width': masked_image.shape[2],
            'transform': out_transform,
        }
    )

    with rasterio.open(file_path, 'w', **out_profile) as dst:
        dst.write(masked_image)

        if body.fusion_mode == FusionMode.ONLY_CORINE:
            dst.write_colormap(1, request.app.state.corine_cmap)
        else:
            dst.write_colormap(1, request.app.state.osm_cmap)

    log.info(f'Finished creating segmentation for {body}')

    return GeoTiffResponse(
        file_path,
        media_type='image/geotiff',
        filename='segmentation.tiff',
        background=BackgroundTask(unlink),
    )


def __analyse(body: LulcWorkUnit, request: Request) -> AnalysisResult:
    def hashable_labels(labels: List[LabelDescriptor]) -> HashableDict:
        labels_dict = dict({i.name: i for i in labels})
        return HashableDict(labels_dict)

    bbox = BBox(bbox=body.area_coords, crs=SCRS.WGS84)
    buffered_bbox = (
        to_utm_bbox(bbox).buffer(request.app.state.edge_smoothing_buffer, relative=False).transform(crs=SCRS.WGS84)
    )
    buffered_bbox = buffered_bbox.lower_left + buffered_bbox.upper_right

    labels, (h, w) = analyse(
        imagery_store=request.app.state.imagery_store,
        osm=request.app.state.osm,
        inference_session=request.app.state.inference_session,
        tx=request.app.state.tx,
        osm_lulc_mapping=hashable_labels(request.app.state.osm_labels),
        corine_lulc_mapping=hashable_labels(request.app.state.corine_labels),
        threshold=body.threshold,
        area_coords=buffered_bbox,
        start_date=body.start_date.isoformat(),
        end_date=body.end_date.isoformat(),
        fusion_mode=body.fusion_mode,
    )

    return AnalysisResult(labels, h, w, buffered_bbox, bbox.geometry)


def hashable_osm_labels(labels: List[LabelDescriptor]) -> HashableDict:
    labels_dict = dict([(d.name, d.osm_filter) for d in labels if d.osm_filter is not None])
    return HashableDict(labels_dict)


@segment.get('/describe', description='Return semantic segmentation labels dictionary')
async def segment_describe(request: Request) -> LabelResponse:
    return LabelResponse(
        osm={label.name: label for label in request.app.state.osm_labels},
        corine={label.name: label for label in request.app.state.corine_labels},
    )


app = FastAPI(lifespan=configure_dependencies)
app.include_router(segment)
app.include_router(health)

if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))

    log.info('Starting LULC Utility')
    uvicorn.run(
        'api:app',
        host='0.0.0.0',
        port=int(os.getenv('LULC_UTILITY_API_PORT', 8000)),
        root_path=os.getenv('ROOT_PATH', '/'),
        log_config=log_config,
        log_level=log_level.lower(),
        workers=int(os.getenv('LULC_UVICORN_WORKERS', 1)),
    )
