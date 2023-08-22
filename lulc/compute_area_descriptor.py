from pathlib import Path

import hydra
from omegaconf import DictConfig

from lulc.data.grid import GridCalculator


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
    compute_area_descriptor()
