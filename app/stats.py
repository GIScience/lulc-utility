from typing import Callable, Optional, Tuple

import numpy as np
from onnxruntime import InferenceSession
from scipy.special import softmax
from scipy.stats import entropy as s_entropy

from lulc.ops.imagery_store_operator import ImageryStore


def tta_uncertainty(
    imagery_store: ImageryStore,
    inference_session: InferenceSession,
    tx: Callable,
    area_coords: Tuple[float, float, float, float],
    start_date: Optional[str],
    end_date: str,
):
    """
    Calculates uncertainty based on prediction variance and Shannon entropy using the test time augmentation approach:
    https://arxiv.org/html/2402.06892v1

    :param imagery_store: operator used to communicate with the remote sensing imagery store
    :param inference_session: ONNX inference session
    :param tx: data transformation and adjustment pipeline
    :param area_coords: coordinates of the prediction target area
    :param start_date: Lower bound (inclusive) of remote sensing data acquisition date (UTC)
    :param end_date: Upper bound (inclusive) of remote sensing data, OSM reference data acquisition date (UTC)
    :return:
    """
    imagery, imagery_size = imagery_store.imagery(area_coords, start_date, end_date)
    imagery = tx({'imagery': imagery})
    image = imagery['imagery'][0]

    proba = tta(image, inference_session)

    mean_proba = np.mean(proba, axis=0)
    entropy = s_entropy(mean_proba, base=mean_proba.shape[0], axis=0)
    variance = np.var(proba, axis=0)
    uncertainty = np.max(variance, axis=0)

    return np.array([uncertainty, entropy]), imagery_size


def tta(image: np.ndarray, inference_session: InferenceSession):
    def aux(x):
        logits = inference_session.run(output_names=None, input_feed={'imagery': x[np.newaxis, ...]})[0][0]
        return softmax(logits, axis=0)

    proba = []

    for k in range(1, 5):
        image = np.rot90(image, k=1, axes=(1, 2))

        proba_part = aux(image)
        proba_part = np.rot90(proba_part, k=-k, axes=(1, 2))
        proba.append(proba_part)

        flipped_image = image[:, ::-1, :]
        proba_part = aux(flipped_image)
        proba_part = proba_part[:, ::-1, :]
        proba_part = np.rot90(proba_part, k=-k, axes=(1, 2))
        proba.append(proba_part)

    return np.array(proba)
