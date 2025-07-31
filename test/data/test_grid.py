from pathlib import Path

import geopandas as gpd
from shapely import Polygon

from lulc.data.grid import GridCalculator

ROOT = Path(__file__).parent.parent.parent
aoi_gdf = gpd.read_file(ROOT / 'data/regierungsbezirk_karlsruhe.geojson').query('osm_id == 285864')
land_mask = gpd.read_file(ROOT / 'data/world_generalized.geojson')
default_grid_calculator = GridCalculator(
    aoi_gdf=aoi_gdf,
    aoi_id_col='osm_id',
    split_mode='osm',
    start_date='2025-05-01',
    end_date='2025-09-30',
    zoom_level=11,
    bbox_size_m=(25000, 25000),
    land_mask=land_mask,
)


def test_check_land_coverage_passes():
    bbox_geom = Polygon.from_bounds(9.71, 52.35, 9.77, 52.38)
    assert default_grid_calculator.check_land_coverage(bbox_geom)


def test_check_land_coverage_fails():
    bbox_geom = Polygon.from_bounds(-0.01, -0.01, 0.01, 0.01)
    assert not default_grid_calculator.check_land_coverage(bbox_geom)


def test_splitter():
    gridded_area = default_grid_calculator.split()
    assert len(gridded_area) == 5
