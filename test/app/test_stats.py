from unittest.mock import Mock, patch

import numpy as np

from app.stats import tta, tta_uncertainty


def test_tta_uncertainty():
    def tx(x):
        return x

    mocked_session = Mock()
    mocked_session.run.return_value = np.random.random((1, 1, 13, 300, 300))

    mocked_imagery_store = Mock()
    mocked_imagery_store.imagery.return_value = np.random.random((1, 13, 300, 300)), (300, 300)

    result = tta_uncertainty(
        mocked_imagery_store, mocked_session, area_coords=(0, 0, 0, 0), tx=tx, start_date=None, end_date='2024-01-01'
    )
    assert result[0].shape == (2, 300, 300)
    assert 0 <= result[0].max() <= 1


@patch('app.stats.softmax')
def test_tta(softmax_mock):
    def softmax_aux(x, axis):
        return x

    def run_aux(output_names, input_feed):
        return input_feed['imagery'][np.newaxis, ...]

    test_image = np.zeros((2, 10, 10))
    test_image[..., 0] = 1

    mocked_session = Mock()
    mocked_session.run.side_effect = run_aux

    softmax_mock.side_effect = softmax_aux
    result = tta(test_image, mocked_session)

    np.testing.assert_equal(test_image, result.mean(axis=0))
