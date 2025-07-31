import logging
import uuid
from functools import partial
from typing import Tuple, Optional
import warnings

import geopandas as gpd
import pandas as pd
from sentinelhub import OsmSplitter, CRS, UtmZoneSplitter
from shapely.geometry import MultiPolygon, Polygon

DESCRIPTOR_COLUMNS = ['uuid', 'min_x', 'min_y', 'max_x', 'max_y', 'start_date', 'end_date']

log = logging.getLogger(__name__)


class GridCalculator:
    def __init__(
        self,
        aoi_gdf: gpd.GeoDataFrame,
        aoi_id_col: str,
        split_mode: str,
        start_date: str,
        end_date: str,
        zoom_level: int,
        bbox_size_m: Tuple[int, int],
        land_mask: gpd.GeoDataFrame,
        land_area_share: float = 0.75,
        sampling_frac: Optional[float] = None,
    ):
        self.aoi_gdf = aoi_gdf
        self.aoi_id_col = aoi_id_col
        self.start_date = start_date
        self.end_date = end_date
        self.land_mask = land_mask
        self.land_area_share = land_area_share
        self.sampling_frac = sampling_frac

        if split_mode.lower() == 'osm':
            self.splitter = partial(OsmSplitter, zoom_level=zoom_level)
        elif split_mode.lower() == 'utm':
            self.splitter = partial(UtmZoneSplitter, bbox_size=bbox_size_m)
        else:
            raise ValueError(f'Split mode {split_mode} not available')

    def check_land_coverage(self, bbox_geom: Polygon) -> bool:
        possible_matches = list(self.land_mask.sindex.intersection(bbox_geom.bounds))
        relevant_land = self.land_mask.iloc[possible_matches]
        with warnings.catch_warnings():
            # Accept warnings about calculating area in a geographic CRS, because we are just interested in the proportion
            warnings.filterwarnings(action='ignore', message=".*Results from 'area' are likely incorrect.*")
            intersection = relevant_land.intersection(bbox_geom).area.sum()
            land_ratio = intersection / bbox_geom.area if bbox_geom.area > 0 else 0
        return land_ratio >= self.land_area_share

    def split(self) -> pd.DataFrame:
        self.aoi_gdf['geometry'] = [
            MultiPolygon([f]) if isinstance(f, Polygon) else f for f in self.aoi_gdf['geometry']
        ]
        osm_splitter = self.splitter(shape_list=self.aoi_gdf.geometry.to_list(), crs=CRS.WGS84)

        descriptor = []
        geometry = []
        for bbox in osm_splitter.bbox_list:
            bbox = bbox.transform(CRS.WGS84)

            if not self.check_land_coverage(bbox.geometry):
                continue  # skip tiles with too little land coverage

            coords = [str(uuid.uuid4())] + list(tuple(bbox)) + [self.start_date, self.end_date]
            descriptor.append(coords)
            geometry.append(bbox.geometry)

        df = gpd.GeoDataFrame(descriptor, columns=DESCRIPTOR_COLUMNS, geometry=geometry, crs=str(osm_splitter.crs))
        if self.sampling_frac:
            df = df.sample(frac=self.sampling_frac)
        return df
