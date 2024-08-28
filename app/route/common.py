import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from pydantic import BaseModel, Field, confloat, model_validator
from rasterio.crs import CRS
from rasterio.mask import mask
from shapely import Polygon
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import FileResponse, Response

from app.process import FusionMode


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


class ImageResponse(Response):
    media_type = 'image/png'


class GeoTiffResponse(FileResponse):
    media_type = 'image/geotiff'


@dataclass
class ProcessingResult:
    payload: np.ndarray
    height: int
    width: int
    area_coords: Tuple[float, float, float, float]
    clip_geometry: Polygon


def __compute_raster_response(result: ProcessingResult, body: LulcWorkUnit, request: Request, write_colormap=True):
    file_uuid = uuid.uuid4()
    file_path = Path(f'/tmp/{file_uuid}.tiff')

    def unlink():
        file_path.unlink()

    with rasterio.open(
        file_path,
        mode='w+',
        driver='GTiff',
        height=result.height,
        width=result.width,
        count=1 if len(result.payload.shape) == 2 else result.payload.shape[0],
        dtype=result.payload.dtype,
        crs=CRS.from_string('EPSG:4326'),
        nodata=None,
        transform=rasterio.transform.from_bounds(*result.area_coords, width=result.width, height=result.height),
    ) as dst:
        if len(result.payload.shape) == 2:
            dst.write(result.payload, 1)
        else:
            for ch_idx, ch_data in enumerate(result.payload, start=1):
                dst.write(ch_data, ch_idx)

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

        if write_colormap:
            if body.fusion_mode == FusionMode.ONLY_CORINE:
                dst.write_colormap(1, request.app.state.corine_cmap)
            else:
                dst.write_colormap(1, request.app.state.osm_cmap)

    return GeoTiffResponse(
        file_path,
        media_type='image/geotiff',
        filename=f'{file_uuid}.tiff',
        background=BackgroundTask(unlink),
    )
