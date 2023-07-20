import tempfile
from pathlib import Path

import pytest

from lulc.ops.osm_operator import OhsomeOps


@pytest.mark.external
def test_ohsome_fetch():
    test_coords = (8.674092, 49.417479, 8.778598, 49.430438)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        ops = OhsomeOps(cache_dir=temp_dir)

        osm_lulc = {
            'urban': 'landuse=residential or landuse=industrial',
            'forest': 'landuse=forest or natural=wood'
        }

        result = ops.labels(test_coords, '2020-06-30', osm_lulc, (300, 300))
        assert ['urban', 'forest'], list(result.keys())
        assert (300, 300) == result['urban'].shape, 'Shapes should match'

