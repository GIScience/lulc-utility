import logging
import os
from pathlib import Path

import yaml

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level, force=True)

config_dir = os.getenv('LULC_UTILITY_APP_CONFIG_DIR', str(Path('conf').absolute()))
log_config = f'{config_dir}/logging/app/logging.yaml'
with open(log_config) as file:
    logging.config.dictConfig(yaml.safe_load(file))

# The following libraries have overwhelmingly verbose debug logs and defeat the purpose of our own debug logs
override_log_level = 'INFO' if log_level == 'DEBUG' else log_level
logging.getLogger('bravado').setLevel(override_log_level)
logging.getLogger('bravado_core').setLevel(override_log_level)
logging.getLogger('requests_oauthlib.oauth2_session').setLevel(override_log_level)
logging.getLogger('swagger_spec_validator').setLevel(override_log_level)
logging.getLogger('urllib3.connectionpool').setLevel(override_log_level)

# onnx logs in the C++ implementation must be exposed specifically and have different config than the default logger
# They can also be overwhelming even at the INFO level, so set them separately
onnx_log_level_mapping = {
    'DEBUG': 0,
    'INFO': 1,
    'WARNING': 2,
    'ERROR': 3,
    'CRITICAL': 4,
}
onnx_log_level = onnx_log_level_mapping[os.getenv('LOG_LEVEL_ONNX', 'WARNING').upper()]
