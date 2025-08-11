import logging.config
import os
import yaml
from pathlib import Path
from typing import Tuple

import contextily as cx
import hydra
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
from geopy.geocoders import Nominatim
from matplotlib import pyplot as plt
from shapely import wkt
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from lulc.data.grid import GridCalculator

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging/app/logging.yaml'
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='area_descriptor')
def compute_area_descriptor(cfg: DictConfig) -> None:
    """
    Compute the area data descriptor by splitting the AOI into a grid.

    Saves to `cfg.area.output_dir`:
     - a csv containing a GeoDataFrame representation of the split (gridded) area, with a copy of each cell for each
       timeframe included in cfg.area_descriptor.timeframes
     - a png visualisation of the split (gridded) area

    :param cfg: loaded area descriptor configuration
    :return: The area descriptor is saved as a CSV file and a visualization is generated as a PNG file.
    """
    output_dir = Path(cfg.area.output_dir)

    log.info('Retrieving area')
    land_mask = gpd.read_file('data/world_generalized.geojson')
    aoi_source, out_name = retrieve_area(cfg)

    log.info(f'Computing area descriptors for {out_name}')
    aoi_id_col = getattr(cfg.area, 'aoi_id_col', 'osm_id')

    targets_specified = False
    if cfg.area.target_aoi_ids:
        aoi_gdf = aoi_source[aoi_source[aoi_id_col].isin(cfg.area.target_aoi_ids)].copy()
        if not aoi_gdf.empty:
            log.info('Filtering AOI by provided target_aoi_ids.')
            targets_specified = True
        else:
            log.info('None of the target_aoi_ids are present in the AOI object. Ignoring target_aoi_ids.')
            aoi_gdf = aoi_source.copy()
    else:
        aoi_gdf = aoi_source.copy()

    log.info('Computing area descriptor')

    dfs = []
    prog = tqdm(cfg.area.timeframes)

    with ProcessPoolExecutor() as executor:
        future_area_descriptor = {
            executor.submit(build_grid_cells, start_date, end_date, aoi_gdf, aoi_id_col, land_mask, cfg): (
                start_date,
                end_date,
            )
            for start_date, end_date in cfg.area.timeframes
        }
        for area_descriptor in as_completed(future_area_descriptor):
            start_date, end_date = future_area_descriptor[area_descriptor]
            prog.set_description(f'Computing area descriptor ({start_date}-{end_date})')
            dfs.append(area_descriptor.result())
            prog.update(1)

    prog.set_description('Computing area descriptor (completed)')
    prog.close()

    df = pd.concat(dfs)

    aoi_name = f'area_{out_name.lower()}'
    if targets_specified:
        aoi_name += '_clipped'

    descriptor_png = output_dir / f'{aoi_name}.png'
    log.info(f'Persisting descriptor visualization: {descriptor_png}')
    ax = df.plot(figsize=(25, 25), alpha=0.05, edgecolor='black', lw=0.7)
    cx.add_basemap(ax, crs=df.crs, source=cx.providers.CartoDB.Positron)
    plt.title(f'Area Descriptor for: {out_name.title()}')
    plt.savefig(descriptor_png, bbox_inches='tight', pad_inches=0)
    plt.close()

    descriptor_csv = output_dir / f'{aoi_name}.csv'
    df.to_csv(descriptor_csv, index=False)

    log.info(f'Area descriptor created. Make sure to add {aoi_name[5:]} to cfg.data.descriptor.area.')


def retrieve_area(cfg: DictConfig) -> Tuple[gpd.GeoDataFrame, str]:
    """Retrieve AOI either from a file or by geocoding a name."""
    config_path = f'{HydraConfig.get().job.config_name}.yaml'
    aoi_file = getattr(cfg.area, 'aoi_file', None)
    aoi_name = getattr(cfg.area, 'aoi_name', None)

    if not aoi_file and not aoi_name:
        raise ValueError(
            f"Neither 'aoi_file' nor 'aoi_name' is provided in the configuration. Edit configuration in {config_path}."
        )

    try:
        if aoi_file:
            aoi_gdf = gpd.read_file(aoi_file)
            out_name = Path(aoi_file).stem.replace(' ', '_')
            return aoi_gdf, out_name

        else:
            return geocode_aoi(aoi_name)

    except FileNotFoundError:
        raise FileNotFoundError(f'{aoi_file} not found. Edit configuration in {config_path}.')

    except Exception:
        raise RuntimeError(f'Error retrieving AOI. Check config file {config_path}.')


def geocode_aoi(aoi_name: str) -> Tuple[gpd.GeoDataFrame, str]:
    """Geocode the AOI name to retrieve its object."""
    log.info(f'Searching for location: {aoi_name}')
    geolocator = Nominatim(user_agent='ClimateAction/LULC')
    locations = geolocator.geocode(aoi_name, geometry='wkt', exactly_one=False)

    if not locations:
        raise ValueError(f'No locations found for "{aoi_name}".')

    if len(locations) > 1:
        log.info(f'Multiple locations found for "{aoi_name}":')
        for idx, loc in enumerate(locations):
            print(f'  [{idx}] {getattr(loc, "address", str(loc))}')
        while True:
            try:
                selection = int(input(f'Select index (0-{len(locations)-1}): '))
                if 0 <= selection < len(locations):
                    location = locations[selection]
                    break
                log.warning('Invalid input.')
            except ValueError:
                log.warning('Invalid input. Enter a number.')
    else:
        location = locations[0]

    location_id = getattr(location, 'osm_id', None)
    location_name = str(location)
    location_address = getattr(location, 'address', None)
    log.info(f'Found: {location_address}; OSM ID {location_id}.')

    aoi_gdf = gpd.GeoDataFrame(
        {
            'osm_id': [location_id],
            'name': [location_name],
            'geometry': [wkt.loads(location.raw['geotext'])],
        },
        crs='EPSG:4326',
    )
    out_name = aoi_name.strip().split(',')[0].replace(' ', '_')
    return aoi_gdf, out_name


def build_grid_cells(
    start_date: str,
    end_date: str,
    aoi_gdf: gpd.GeoDataFrame,
    aoi_id_col: str,
    land_mask: gpd.GeoDataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Generates entries (grid cells) of the area descriptor for the given time period."""
    calculator = GridCalculator(
        aoi_gdf=aoi_gdf,
        aoi_id_col=aoi_id_col,
        split_mode=cfg.area.split_mode,
        start_date=start_date,
        end_date=end_date,
        zoom_level=cfg.area.split_params.zoom_level,
        bbox_size_m=cfg.area.split_params.bbox_size_m,
        sampling_frac=cfg.area.sampling_frac,
        land_mask=land_mask,
        land_area_share=cfg.area.land_area_share,
    )
    return calculator.split()


if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))
    log.info('Computing area descriptor')
    compute_area_descriptor()
