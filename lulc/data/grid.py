import logging
import uuid
from functools import partial
from typing import List, Tuple, Optional

import geopandas as gpd
import pandas as pd
from sentinelhub import OsmSplitter, CRS, UtmZoneSplitter
from shapely.geometry import MultiPolygon, Polygon

DESCRIPTOR_COLUMNS = ['uuid', 'min_x', 'min_y', 'max_x', 'max_y', 'start_date', 'end_date']

log = logging.getLogger(__name__)


class GridCalculator:

    def __init__(self, nuts_source: str, nuts_id_col: str, split_mode: str, start_date: str, end_date: str,
                 zoom_level: int, bbox_size_m: Tuple[int, int], sampling_frac: Optional[float] = None):
        self.gdf = gpd.read_file(nuts_source)
        self.nuts_id_col = nuts_id_col
        self.start_date = start_date
        self.end_date = end_date
        self.sampling_frac = sampling_frac

        if split_mode.lower() == 'osm':
            self.splitter = partial(OsmSplitter, zoom_level=zoom_level)
        elif split_mode.lower() == 'utm':
            self.splitter = partial(UtmZoneSplitter, bbox_size=bbox_size_m)
        else:
            raise ValueError(f'Split mode {split_mode} not available')

    def split(self, target_nuts_ids: List[str]) -> pd.DataFrame:
        sub_gdf = self.gdf[self.gdf[self.nuts_id_col].isin(target_nuts_ids)].copy()
        sub_gdf['geometry'] = [MultiPolygon([f]) if isinstance(f, Polygon) else f for f in sub_gdf['geometry']]
        osm_splitter = self.splitter(shape_list=sub_gdf.geometry.to_list(), crs=CRS.WGS84)

        descriptor = []
        geometry = []
        for bbox in osm_splitter.bbox_list:
            bbox = bbox.transform(CRS.WGS84)
            coords = [str(uuid.uuid4())] + list(tuple(bbox)) + [self.start_date, self.end_date]
            descriptor.append(coords)
            geometry.append(bbox.geometry)

        df = gpd.GeoDataFrame(descriptor, columns=DESCRIPTOR_COLUMNS, geometry=geometry, crs=str(osm_splitter.crs))
        if self.sampling_frac:
            df = df.sample(frac=self.sampling_frac)
        return df
