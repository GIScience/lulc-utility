import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.location import Location
from omegaconf import DictConfig, ListConfig
from pyogrio.errors import DataSourceError
from shapely import wkt

log = logging.getLogger(__name__)


def retrieve_area(cfg: DictConfig) -> Tuple[gpd.GeoDataFrame, str]:
    """Retrieve AOI(s) either from a file or by geocoding."""
    aoi_file = getattr(cfg, 'aoi_file', None)
    aoi_geocode = getattr(cfg, 'aoi_geocode', None)

    if aoi_file:
        try:
            aoi_gdf = gpd.read_file(aoi_file)
        except DataSourceError:
            raise FileNotFoundError(
                f'Specified AOI file {aoi_file} not found. Check path in configuration or choose to geocode by providing "aoi_geocode" instead.'
            )
    elif aoi_geocode:
        if isinstance(aoi_geocode, ListConfig):
            aoi_geocode = list(aoi_geocode)
        elif isinstance(aoi_geocode, str):
            aoi_geocode = [aoi_geocode]
        aoi_gdf = geocode_area(aoi_geocode)
    else:
        raise ValueError('Neither "aoi_file" nor "aoi_geocode" is provided in the configuration.')

    out_name = extract_area_name(cfg)
    return aoi_gdf, out_name


def geocode_area(aoi_geocode: list[str]) -> gpd.GeoDataFrame:
    """Geocode requested areas based on provided geocode(s) and retrieve geodata."""
    locations_gdf = geolocating(aoi_geocode)

    locations_info = []
    for row in locations_gdf.itertuples(index=False):
        locations_info.append(f'  - {row.name} ({row.admintype}); OSM ID {row.osm_id}')
    log.info(f'Geocoding completed. Following location(s) retrieved successfully:\n{"\n".join(locations_info)}')

    while True:
        user_input = input('Do you want to proceed with all retrieved locations? (y/n): ').strip().lower()
        if user_input in ['y', 'yes', '']:
            break
        elif user_input in ['n', 'no']:
            log.info('Aborting as per user request.')
            sys.exit(0)
        else:
            log.warning('Invalid input. Yes or no, abort with Ctrl+C.')

    return locations_gdf


def geolocating(aoi_geocode: list[str]) -> gpd.GeoDataFrame:
    """Use Nominatim to geolocate requested areas."""
    locations_list = []

    geolocator = Nominatim(user_agent='ClimateAction/LULCUtility')

    for area in aoi_geocode:
        results = geolocator.geocode(area, geometry='wkt', exactly_one=False, language='en')

        if not results:
            raise ValueError(f'No location found for "{area}". Check the geocode in the configuration.')

        if len(results) > 1:
            results_filtered = [loc for loc in results if get_loc_info(loc)['featuretype'] == 'relation']
            if len(results_filtered) == 1:
                location = results_filtered[0]
            elif len(results_filtered) > 1:
                location = location_selector(results_filtered, area)
            else:
                raise ValueError(
                    f'Multiple locations found for "{area}", but none is a relation. Please further specify "aoi_geocode" in the configuration.'
                )
        else:
            location = results[0]

        loc_data = get_loc_info(location)

        area_gdf = gpd.GeoDataFrame(
            {
                'osm_id': [loc_data['id']],
                'name': [loc_data['name']],
                'admintype': [loc_data['admintype']],
                'featuretype': [loc_data['featuretype']],
                'geometry': [wkt.loads(location.raw['geotext'])],
            },
            crs='EPSG:4326',
        )

        locations_list.append(area_gdf)

    return gpd.GeoDataFrame(data=pd.concat(locations_list, ignore_index=True), crs='EPSG:4326')


def location_selector(results: list[Location], area: str) -> Location:
    """Let user select correct location from multiple geocoding results."""
    locations_info = []
    for idx, loc in enumerate(results):
        loc_data = get_loc_info(loc)
        locations_info.append(f'  [{idx}] {loc_data["name"]} ({loc_data["admintype"]}); OSM ID {loc_data["id"]}')
    log.info(f'Multiple locations found for "{area}":\n{"\n".join(locations_info)}')

    warning_msg = 'Invalid input. Select one of the shown indexes or abort with Ctrl+C.'
    while True:
        selection = input(f'Select index (0-{len(results) - 1}): ')
        try:
            selection = int(selection)
        except ValueError:
            log.warning(warning_msg)
            continue

        if 0 <= selection < len(results):
            location = results[selection]
            break
        else:
            log.warning(warning_msg)

    return location


def get_loc_info(location: Location) -> Dict[str, str]:
    """Extract location information from a geopy Location object."""
    return {
        'id': getattr(location, 'osm_id', location.raw.get('osm_id', None)),
        'name': getattr(location, 'address', location.raw.get('name', str(location))),
        'admintype': location.raw.get('addresstype', 'unknown'),
        'featuretype': location.raw.get('osm_type', 'unknown'),
    }


def extract_area_name(cfg: DictConfig) -> str:
    """Determine and return the formatted area name for outputs based on the configuration."""
    area_source = getattr(cfg, 'aoi_file', getattr(cfg, 'aoi_geocode', None))
    aoi_name = getattr(cfg, 'aoi_name', area_source)

    if isinstance(aoi_name, ListConfig):
        names = [format_name(name) for name in aoi_name]
        return '_'.join(names)
    else:
        return format_name(aoi_name)


def format_name(name: str) -> str:
    """Format the area name for output naming."""
    return Path(name).stem.strip().split(',')[0].replace(' ', '_').lower()
