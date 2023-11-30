import logging.config
import os
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig

from lulc.data.grid import GridCalculator

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging/app/logging.yaml'
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='area_descriptor')
def compute_area_descriptor(cfg: DictConfig) -> None:
    """
    Compute the area data descriptor by splitting the chosen NUTS regions into a grid.

    :param cfg: underlying Hydra configuration
    :return:
    """

    calculator = GridCalculator(nuts_source=cfg.area.nuts_source,
                                nuts_id_col=cfg.area.nuts_id_col,
                                split_mode=cfg.area.split_mode,
                                start_date=cfg.area.timeframe.start_date,
                                end_date=cfg.area.timeframe.end_date,
                                output_dir=Path(cfg.area.output_dir),
                                zoom_level=cfg.area.split_params.zoom_level,
                                bbox_size_m=cfg.area.split_params.bbox_size_m)
    calculator.split(target_nuts_ids=cfg.area.target_nuts_ids)


if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))
    log.info('Computing area desciptor')
    compute_area_descriptor()
