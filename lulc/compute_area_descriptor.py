import logging.config
import os
from pathlib import Path

import contextily as cx
import hydra
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

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
    output_dir = Path(cfg.area.output_dir)

    dfs = []
    prog = tqdm(cfg.area.timeframes)
    for i, (start_date, end_date) in enumerate(prog):
        prog.set_description(f'Computing area descriptors ({start_date}-{end_date})')
        calculator = GridCalculator(
            nuts_source=cfg.area.nuts_source,
            nuts_id_col=cfg.area.nuts_id_col,
            split_mode=cfg.area.split_mode,
            start_date=start_date,
            end_date=end_date,
            zoom_level=cfg.area.split_params.zoom_level,
            bbox_size_m=cfg.area.split_params.bbox_size_m,
            sampling_frac=cfg.area.sampling_frac,
        )
        df = calculator.split(target_nuts_ids=cfg.area.target_nuts_ids)
        dfs.append(df)

        if i == len(prog) - 1:
            prog.set_description('Computing area descriptors (completed)')

    df = pd.concat(dfs)

    descriptor_png = output_dir / 'area_output.png'
    log.info(f'Persisting descriptor visualization: {descriptor_png}')
    ax = df.plot(figsize=(25, 25), alpha=0.1, edgecolor='black', lw=0.7)
    plt.title(f'NUTS: {cfg.area.nuts_id_col}')
    cx.add_basemap(ax, crs=df.crs, source=cx.providers.CartoDB.Positron)
    plt.savefig(str(descriptor_png), bbox_inches='tight', pad_inches=0)
    plt.close()

    descriptor_csv = str(output_dir / 'area_output.csv')
    df.to_csv(descriptor_csv, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))
    log.info('Computing area descriptor')
    compute_area_descriptor()
