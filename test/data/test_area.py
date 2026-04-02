from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from omegaconf import OmegaConf

from lulc.data.area import extract_area_name, format_name, retrieve_area


@pytest.fixture
def vcr_config():
    return {
        'record_mode': 'once',
        'cassette_library_dir': 'test/cassettes',
    }


@pytest.mark.vcr()
def test_retrieve_area_with_aoi_geocode():
    """
    Test retrieve_area when aoi_geocode is provided in config with/without aoi_name.
    Expected behavior: aoi_geocode is used to retrieve the AOI and the result is returned formatted as out_name.
    """
    geocode = 'Heidelberg, Germany'

    cfg_wo = OmegaConf.create({'aoi_geocode': geocode})
    cfg_w = OmegaConf.create({'aoi_geocode': geocode, 'aoi_name': 'City HD'})

    with patch('builtins.input', return_value=''):
        aoi_gdf_wo, out_name_wo = retrieve_area(cfg_wo)
        aoi_gdf_w, out_name_w = retrieve_area(cfg_w)

    assert out_name_wo == 'heidelberg'
    assert type(aoi_gdf_wo) is gpd.GeoDataFrame
    assert len(aoi_gdf_wo) == 1

    assert out_name_w == 'city_hd'
    assert type(aoi_gdf_w) is gpd.GeoDataFrame
    assert len(aoi_gdf_w) == 1


@pytest.mark.vcr()
def test_retrieve_area_with_aoi_geocode_list():
    """
    Test retrieve_area when aoi_geocode is provided in config as list with/without aoi_name.
    Expected behavior: aoi_geocode is used to retrieve multiple AOIs and the result is returned formatted as out_name.
    """
    geocodes = [
        'Heidelberg, Germany',
        'Mannheim, Germany',
    ]

    cfg_wo = OmegaConf.create({'aoi_geocode': geocodes})
    cfg_w = OmegaConf.create({'aoi_geocode': geocodes, 'aoi_name': 'Cities HD & MA'})

    with patch('builtins.input', return_value=''):
        aoi_gdf_wo, out_name_wo = retrieve_area(cfg_wo)
        aoi_gdf_w, out_name_w = retrieve_area(cfg_w)

    assert out_name_wo == 'heidelberg_mannheim'
    assert type(aoi_gdf_wo) is gpd.GeoDataFrame
    assert len(aoi_gdf_wo) == 2

    assert out_name_w == 'cities_hd_&_ma'
    assert type(aoi_gdf_w) is gpd.GeoDataFrame
    assert len(aoi_gdf_w) == 2


def test_retrieve_area_with_aoi_file_aoi_geocode():
    """
    Test retrieve_area when aoi_file and aoi_geocode is provided in config with/without aoi_name.
    Expected behavior: aoi_file is read, aoi_geocode ignored and aoi_name is used and returned formatted as out_name.
    """
    path = '/path/to/RB Karlsruhe Germany_123 456_OSM.geojson'
    geocode = '123foo, Atlantis'

    cfg_wo = OmegaConf.create(
        {
            'aoi_file': path,
            'aoi_geocode': geocode,
        }
    )
    cfg_w = OmegaConf.create(
        {
            'aoi_file': path,
            'aoi_geocode': geocode,
            'aoi_name': 'RB DE12',
        }
    )

    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    with patch('lulc.data.area.gpd.read_file', return_value=mock_gdf):
        aoi_gdf_wo, out_name_wo = retrieve_area(cfg_wo)
        aoi_gdf_w, out_name_w = retrieve_area(cfg_w)

    assert out_name_wo == 'rb_karlsruhe_germany_123_456_osm'
    assert aoi_gdf_wo is mock_gdf

    assert out_name_w == 'rb_de12'
    assert aoi_gdf_w is mock_gdf


def test_retrieve_area_with_neither():
    """
    Test retrieve_area when neither aoi_file nor aoi_geocode is provided in config.
    Expected behavior: ValueError is raised with further information.
    """
    cfg = OmegaConf.create({})

    with pytest.raises(ValueError, match='Neither "aoi_file" nor "aoi_geocode" is provided in the configuration.'):
        retrieve_area(cfg)


@pytest.mark.vcr()
def test_retrieve_area_abort():
    """
    Test retrieve_area when user aborts the process after being prompted to confirm the retrieved AOI.
    Expected behavior: Clean SystemExit with error code 0.
    """
    cfg = OmegaConf.create({'aoi_geocode': 'Heidelberg, Germany'})

    with pytest.raises(SystemExit) as exc:
        with patch('builtins.input', return_value='n'):
            retrieve_area(cfg)
    assert exc.value.code == 0


@pytest.mark.vcr()
def test_retrieve_area_with_broken_aoi_geocode():
    """
    Test retrieve_area when aoi_geocode is provided but cannot be geolocated or the locations are not of feature type relation.
    Expected behavior: ValueErrors are raised with further information.
    """
    cfg_noloc = OmegaConf.create({'aoi_geocode': '123foo, Atlantis'})
    cfg_norel = OmegaConf.create({'aoi_geocode': 'Point Nemo'})

    with pytest.raises(
        ValueError,
        match='No location found for "123foo, Atlantis". Check the geocode in the configuration.',
    ):
        retrieve_area(cfg_noloc)

    with pytest.raises(
        ValueError,
        match='Multiple locations found for "Point Nemo", but none is a relation. Please further specify "aoi_geocode" in the configuration.',
    ):
        retrieve_area(cfg_norel)


def test_format_name():
    """Test format_name with various input strings."""
    assert format_name('Heidelberg') == 'heidelberg'
    assert format_name('Mannheim, Germany') == 'mannheim'
    assert format_name('  Baden-Baden  ') == 'baden-baden'
    assert format_name('Bad Dürkheim,DE') == 'bad_dürkheim'
    assert format_name('') == ''
    assert format_name('   ') == ''
    assert format_name(',,') == ''


def test_extract_area_name():
    """Test extract_area_name with various config inputs."""
    cfg = OmegaConf.create({'aoi_geocode': 'Los Angeles'})
    assert extract_area_name(cfg) == 'los_angeles'

    cfg = OmegaConf.create({'aoi_geocode': ['New York, NY, USA', 'Boston,USA']})
    assert extract_area_name(cfg) == 'new_york_boston'

    cfg = OmegaConf.create({'aoi_geocode': ['New York, NY, USA', 'Boston,USA'], 'aoi_name': 'North East'})
    assert extract_area_name(cfg) == 'north_east'

    cfg = OmegaConf.create({'aoi_file': '/path/to/Chicago.geojson'})
    assert extract_area_name(cfg) == 'chicago'

    cfg = OmegaConf.create({'aoi_file': '/path/to/Heidelberg.geojson', 'aoi_name': 'Heidelberg City'})
    assert extract_area_name(cfg) == 'heidelberg_city'
