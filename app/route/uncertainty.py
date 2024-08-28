import io

import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException
from sentinelhub import BBox, CRS as SCRS, to_utm_bbox
from starlette.requests import Request

from app.process import FusionMode
from app.route.common import LulcWorkUnit, __compute_raster_response, ImageResponse, ProcessingResult
from app.stats import tta_uncertainty

router = APIRouter(prefix='/uncertainty')


@router.post(
    '/',
    description='Run the semantic segmentation algorithm against a test-time augmentation step to calculate uncertainty metrics',
)
async def uncertainty_compute(body: LulcWorkUnit, request: Request):
    if body.fusion_mode == FusionMode.ONLY_MODEL:
        result = __uncertainty(body, request)
        return __compute_raster_response(result, body, request, write_colormap=False)
    else:
        raise HTTPException(
            status_code=400,
            detail=f'Fusion mode {body.fusion_mode} does not support an uncertainty calculation procedure',
        )


def __uncertainty(body: LulcWorkUnit, request: Request):
    bbox = BBox(bbox=body.area_coords, crs=SCRS.WGS84)
    buffered_bbox = (
        to_utm_bbox(bbox).buffer(request.app.state.edge_smoothing_buffer, relative=False).transform(crs=SCRS.WGS84)
    )
    buffered_bbox = buffered_bbox.lower_left + buffered_bbox.upper_right

    uncertainty, (h, w) = tta_uncertainty(
        imagery_store=request.app.state.imagery_store,
        inference_session=request.app.state.inference_session,
        tx=request.app.state.tx,
        area_coords=buffered_bbox,
        start_date=body.start_date.isoformat(),
        end_date=body.end_date.isoformat(),
    )

    return ProcessingResult(uncertainty, h, w, buffered_bbox, bbox.geometry)


@router.post(
    '/preview',
    description='Run the semantic segmentation algorithm against a test-time augmentation step to calculate uncertainty metrics '
    '(low-resolution, medium-compression image, only a min-max scaled uncertainty layer)',
    response_class=ImageResponse,
)
async def uncertainty_preview(body: LulcWorkUnit, request: Request) -> ImageResponse:
    result = __uncertainty(body, request)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for idx, title in enumerate(['Uncertainty (variance)', 'Entropy (Shannon)']):
        cs = ax[idx].imshow(result.payload[idx])
        ax[idx].set_title(title)
        plt.colorbar(cs, ax=ax[idx])

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='PNG')
    buf.seek(0)

    fig.clear()
    plt.close(fig)

    return ImageResponse(buf.getvalue(), media_type='image/png')
