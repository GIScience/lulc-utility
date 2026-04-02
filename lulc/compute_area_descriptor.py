import logging.config
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import contextily as cx
import geopandas as gpd
import hydra
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from lulc.data.area import retrieve_area
from lulc.data.grid import GridCalculator

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = 'conf/logging.yaml'
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../conf', config_name='config')
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
    output_dir = Path(cfg.train.area.output_dir)

    log.info('Retrieving area')
    land_mask = gpd.read_file('data/world_generalized.geojson')
    aoi_source, out_name = retrieve_area(cfg.train.area)

    log.info(f'Computing area descriptors for {out_name.title()}')
    aoi_id_col = getattr(cfg.train.area, 'aoi_id_col', 'osm_id')

    targets = getattr(cfg.train.area, 'target_aoi_ids', None)
    if targets:
        aoi_gdf = aoi_source[aoi_source[aoi_id_col].isin(cfg.train.area.target_aoi_ids)].copy()
        if not aoi_gdf.empty:
            log.info('Filtered AOI by provided target_aoi_ids.')
        else:
            log.info('None of the target_aoi_ids are present in the AOI object. Ignoring target_aoi_ids.')
            aoi_gdf = aoi_source.copy()
    else:
        aoi_gdf = aoi_source.copy()

    log.info('Computing area descriptor')

    dfs = []
    prog = tqdm(cfg.train.area.timeframes)

    with ProcessPoolExecutor() as executor:
        future_area_descriptor = {
            executor.submit(build_grid_cells, start_date, end_date, aoi_gdf, aoi_id_col, land_mask, cfg.train): (
                start_date,
                end_date,
            )
            for start_date, end_date in cfg.train.area.timeframes
        }
        for area_descriptor in as_completed(future_area_descriptor):
            start_date, end_date = future_area_descriptor[area_descriptor]
            prog.set_description(f'Computing area descriptor ({start_date}-{end_date})')
            dfs.append(area_descriptor.result())
            prog.update(1)

    prog.set_description('Computing area descriptor (completed)')
    prog.close()

    df = pd.concat(dfs)

    descriptor_png = output_dir / f'{out_name}.png'
    log.info(f'Persisting descriptor visualization: {descriptor_png}')
    ax = df.plot(figsize=(25, 25), alpha=0.05, edgecolor='black', lw=0.7)
    aoi_gdf.plot(ax=ax, facecolor='none', edgecolor='black', lw=1)
    cx.add_basemap(ax, crs=df.crs, source=cx.providers.CartoDB.Positron)
    plt.title(f'Area Descriptor for: {out_name.title()}')
    plt.savefig(descriptor_png, bbox_inches='tight', pad_inches=0)
    plt.close()

    descriptor_csv = output_dir / f'{out_name}.csv'
    df.to_csv(descriptor_csv, index=False)
    log.info(f'Area descriptor file saved to {descriptor_csv}.')


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
