import logging.config
import os
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
import uvicorn
import yaml
from fastapi import FastAPI
from hydra import compose
from matplotlib import pyplot as plt
from onnxruntime import InferenceSession, SessionOptions

from app.route import segment, health, uncertainty
from lulc.data.label import resolve_corine_labels, resolve_osm_labels
from lulc.data.tx.array import Normalize, Stack, NanToNum, AdjustShape
from lulc.model.ops.download import NeptuneModelDownload
from lulc.ops.imagery_store_operator import resolve_imagery_store
from lulc.ops.osm_operator import OhsomeOps

config_dir = os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute()))

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_config = f'{config_dir}/logging/app/logging.yaml'
log = logging.getLogger(__name__)

plt.switch_backend('agg')


@asynccontextmanager
async def configure_dependencies(app: FastAPI):
    """
    Initialize all required dependencies and attach them to the FastAPI state.
    Each underlying service utilizes configuration stored in `./conf/`.

    :param app: web application instance
    :return: context manager generator
    """
    log.info('Initialising...')

    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name='config')

    app.state.imagery_store = resolve_imagery_store(cfg.imagery, cache_dir=Path(cfg.cache.dir))
    app.state.osm = OhsomeOps(cache_dir=Path(cfg.cache.dir))

    registry = NeptuneModelDownload(
        model_key=cfg.neptune.model.key,
        project=cfg.neptune.project,
        api_key=cfg.neptune.api_token,
        cache_dir=Path(cfg.cache.dir),
    )

    onnx_model, label_descriptor_version = registry.download_model_version(cfg.serve.model_version)

    app.state.osm_labels = resolve_osm_labels(Path(cfg.data.dir), label_descriptor_version)
    app.state.osm_cmap = dict(enumerate([d.color for d in app.state.osm_labels]))

    app.state.corine_labels = resolve_corine_labels(Path(cfg.data.dir), label_descriptor_version)
    app.state.corine_cmap = dict(enumerate([d.color for d in app.state.corine_labels]))

    options = SessionOptions()
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.enable_mem_reuse = False
    app.state.inference_session = InferenceSession(str(onnx_model), sess_options=options)

    def tx(x):
        x = NanToNum(layers=['s1.tif', 's2.tif'], subset='imagery')(x)
        x = Stack(subset='imagery')(x)
        x = Normalize(subset='imagery', mean=cfg.data.normalize.mean, std=cfg.data.normalize.std)(x)
        return AdjustShape(subset='imagery')(x)

    app.state.tx = tx
    app.state.edge_smoothing_buffer = cfg.serve.edge_smoothing_buffer

    log.info('Initialisation completed')

    yield


app = FastAPI(lifespan=configure_dependencies)
app.include_router(segment.router)
app.include_router(uncertainty.router)
app.include_router(health.router)

if __name__ == '__main__':
    logging.basicConfig(level=log_level.upper())
    with open(log_config) as file:
        logging.config.dictConfig(yaml.safe_load(file))

    log.info('Starting LULC Utility')
    uvicorn.run(
        'api:app',
        host='0.0.0.0',
        port=int(os.getenv('LULC_UTILITY_API_PORT', 8000)),
        root_path=os.getenv('ROOT_PATH', '/'),
        log_config=log_config,
        log_level=log_level.lower(),
        workers=int(os.getenv('LULC_UVICORN_WORKERS', 1)),
    )
