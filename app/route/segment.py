import io
import logging
from typing import Dict, List

import numpy as np
from fastapi import APIRouter
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pydantic import BaseModel, Field
from sentinelhub import BBox, CRS as SCRS, to_utm_bbox, bbox_to_dimensions
from starlette.requests import Request
from webcolors import rgb_to_hex

from app.process import FusionMode, analyse
from app.route.common import GeoTiffResponse, LulcWorkUnit, __compute_raster_response, ProcessingResult, ImageResponse
from lulc.data.label import LabelDescriptor, HashableDict

router = APIRouter(prefix='/segment')

log = logging.getLogger(__name__)


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


@router.post(
    '/',
    description='Run the semantic segmentation algorithm and return a georeferenced raster (GeoTIFF)',
    response_class=GeoTiffResponse,
)
async def segment_compute(body: LulcWorkUnit, request: Request) -> GeoTiffResponse:
    log.info(f'Creating segmentation for {body}')
    result = __analyse(body, request)
    return __compute_raster_response(result, body, request)


def __analyse(body: LulcWorkUnit, request: Request) -> ProcessingResult:
    def hashable_labels(labels: List[LabelDescriptor]) -> HashableDict:
        labels_dict = dict({i.name: i for i in labels})
        return HashableDict(labels_dict)

    bbox = BBox(bbox=body.area_coords, crs=SCRS.WGS84)
    buffered_bbox = (
        to_utm_bbox(bbox).buffer(request.app.state.edge_smoothing_buffer, relative=False).transform(crs=SCRS.WGS84)
    )
    buffered_bbox = buffered_bbox.lower_left + buffered_bbox.upper_right
    bbox_width, bbox_height = bbox_to_dimensions(BBox(bbox=buffered_bbox, crs=SCRS.WGS84), resolution=10)
    log.debug(
        f'Buffered input area by {request.app.state.edge_smoothing_buffer} metres to avoid edge effects.'
        f'The dimensions of the new bounding box are ({bbox_width}x{bbox_height} px)'
    )

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

    return ProcessingResult(labels, h, w, buffered_bbox, bbox.geometry)


@router.post(
    '/preview',
    description='Run the semantic segmentation algorithm and return a preview of the result '
    '(low-resolution, medium-compression image)',
    response_class=ImageResponse,
)
async def segment_preview(body: LulcWorkUnit, request: Request) -> ImageResponse:
    result = __analyse(body, request)

    labels = (
        request.app.state.corine_labels if body.fusion_mode == FusionMode.ONLY_CORINE else request.app.state.osm_labels
    )

    image = np.array([d.color for d in labels], dtype=np.uint8)[result.payload]
    patches = [Patch(color=rgb_to_hex(d.color), label=d.name) for d in labels]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='PNG')
    buf.seek(0)

    fig.clear()
    plt.close(fig)

    log.info(f'Finished creating preview for {body}')
    return ImageResponse(buf.getvalue(), media_type='image/png')


@router.get('/describe', description='Return semantic segmentation labels dictionary')
async def segment_describe(request: Request) -> LabelResponse:
    return LabelResponse(
        osm={label.name: label for label in request.app.state.osm_labels},
        corine={label.name: label for label in request.app.state.corine_labels},
    )
