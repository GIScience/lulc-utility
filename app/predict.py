import logging
from enum import Enum
from functools import lru_cache
from typing import Tuple, Callable

import numpy as np
import numpy.ma as ma
from fastapi import HTTPException
from onnxruntime import InferenceSession
from scipy.special import softmax

from lulc.data.tx.array import ReclassifyMerge
from lulc.ops.exception import OperatorValidationException, OperatorInteractionException
from lulc.ops.imagery_store_operator import ImageryStore
from lulc.ops.osm_operator import OhsomeOps

log = logging.getLogger(__name__)


class FusionMode(Enum):
    ONLY_MODEL = 'only_model'
    ONLY_OSM = 'only_osm'
    FAVOUR_OSM = 'favour_osm'
    FAVOUR_MODEL = 'favour_model'
    MEAN_MIXIN = 'mean_mixin'


@lru_cache(maxsize=32)
def predict(imagery_store: ImageryStore,
            osm: OhsomeOps,
            inference_session: InferenceSession,
            tx: Callable,
            osm_lulc_mapping: dict,
            threshold: float,
            area_coords: Tuple[float, float, float, float],
            start_date: str, end_date: str,
            fusion_mode: FusionMode):
    """
    Run the model inference pipeline:
    - call the external remote sensing imagery store to acquire preprocessed data,
    - apply data transformations to adjust the preprocessed data to model requirements,
    - run the inference session.

    :param osm: ohsome client instance
    :param imagery_store: operator used to communicat with the remote sensing imagery store
    :param inference_session: ONNX inference session
    :param tx: data transformation and adjustment pipeline
    :param osm_lulc_mapping: LULC classes expressed as OSM filters
    :param threshold: Class prediction threshold
    :param area_coords: coordinates of the prediction target area
    :param start_date: Lower bound (inclusive) of remote sensing imagery acquisition date (UTC)
    :param end_date: Upper bound (inclusive) of remote sensing imagery and OSM reference data acquisition date (UTC)
    :param fusion_mode: determine whether and how model data has to be merged with OSM data
    :return: 2D numpy array with most probable classes
    """
    log.debug('Running model inference pipeline')

    try:
        imagery, imager_size = imagery_store.imagery(area_coords, start_date, end_date)
        imagery = tx({'imagery': imagery})
        logits = inference_session.run(output_names=None, input_feed=imagery)[0][0]
        labels = __fusion(osm, osm_lulc_mapping, threshold, area_coords, end_date, fusion_mode, logits)
        log.debug('Model inference pipeline completed')
        return labels, imager_size
    except OperatorValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OperatorInteractionException as e:
        raise HTTPException(status_code=500, detail=str(e))


def __fusion(osm: OhsomeOps,
             osm_lulc_mapping: dict,
             threshold: float,
             area_coords: Tuple[float, float, float, float],
             date: str,
             fusion_mode: FusionMode,
             logits: np.ndarray):
    log.debug(f'Fusing predictions in {fusion_mode} mode')
    pred = softmax(logits, axis=0)

    def masked_argmax(x):
        ma_pred = ma.array(x, mask=x < threshold)
        return ma.argmax(ma_pred, axis=0, keepdims=False, fill_value=0).astype(np.uint8)

    if fusion_mode == fusion_mode.ONLY_MODEL:
        labels = masked_argmax(pred)
    elif fusion_mode == fusion_mode.MEAN_MIXIN:
        osm_labels = osm.labels(area_coords, date, osm_lulc_mapping, pred.shape[-2:])
        osm_labels = np.stack(list(osm_labels.values())).astype(np.uint8)
        background = np.ones(pred.shape[-2:]) - np.any(osm_labels, axis=0)
        osm_labels = np.concatenate([background[np.newaxis, ...], osm_labels])
        pred = np.mean(np.array([osm_labels, pred]), axis=0)
        labels = masked_argmax(pred)
    else:
        osm_labels = osm.labels(area_coords, date, osm_lulc_mapping, pred.shape[-2:])
        osm_labels = ReclassifyMerge()({'y': osm_labels})['y']
        labels = masked_argmax(pred)

        if fusion_mode == FusionMode.FAVOUR_OSM:
            labels = np.where(osm_labels == 0, labels, osm_labels)
        elif fusion_mode == FusionMode.FAVOUR_MODEL:
            labels = np.where(labels == 0, osm_labels, labels)
        elif fusion_mode == FusionMode.ONLY_OSM:
            labels = osm_labels
        else:
            raise ValueError(f'Fusion mode {fusion_mode} not supported')

    log.debug(f'Predictions fused in {fusion_mode} mode')

    return labels
