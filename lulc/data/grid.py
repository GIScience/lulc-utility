import logging
import uuid
from functools import partial
from pathlib import Path
from typing import List, Tuple

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sentinelhub import OsmSplitter, CRS, UtmZoneSplitter
from shapely.geometry import MultiPolygon, Polygon

DESCRIPTOR_COLUMNS = ['uuid', 'min_x', 'min_y', 'max_x', 'max_y', 'start_date', 'end_date']

log = logging.getLogger(__name__)


class GridCalculator:

    def __init__(self, nuts_source: str, nuts_id_col: str, split_mode: str, start_date: str, end_date: str,
                 output_dir: Path, zoom_level: int, bbox_size_m: Tuple[int, int]):
        self.gdf = gpd.read_file(nuts_source)
        self.nuts_id_col = nuts_id_col
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir

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

        log.info('Calculating grid geometries')
        descriptor = []
        geometry = []
        for bbox in osm_splitter.bbox_list:
            bbox = bbox.transform(CRS.WGS84)
            coords = [str(uuid.uuid4())] + list(tuple(bbox)) + [self.start_date, self.end_date]
            descriptor.append(coords)
            geometry.append(bbox.geometry)

        df = gpd.GeoDataFrame(descriptor, columns=DESCRIPTOR_COLUMNS, geometry=geometry, crs=str(osm_splitter.crs))

        descriptor_png = self.output_dir / 'area_output.png'
        if not descriptor_png.exists():
            log.info(f'Persisting descriptor visualization: {descriptor_png}')
            ax = df.plot(figsize=(25, 25), alpha=0.3, edgecolor='black', lw=0.7)
            plt.title(f'NUTS: {target_nuts_ids}, start_date: ${self.start_date}, end_date: ${self.end_date}')
            cx.add_basemap(ax, crs=df.crs, source=cx.providers.CartoDB.Positron)
            plt.savefig(str(descriptor_png), bbox_inches='tight', pad_inches=0)
            plt.close()

        return df
