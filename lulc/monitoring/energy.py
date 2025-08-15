import logging
from tempfile import TemporaryDirectory

from neptune import Run
from pyJoules.device import DeviceFactory
from pyJoules.energy_meter import EnergyMeter
from pyJoules.handler.pandas_handler import PandasHandler

log = logging.getLogger(__name__)


class EnergyContext:
    def __init__(self, experiment: Run, init_tag='init', enable_tracking=False):
        self.experiment = experiment
        self.init_tag = init_tag
        self.handler = PandasHandler()
        self.enable_tracking = enable_tracking

        self.readable_devices = []

        for device in DeviceFactory.create_devices():
            try:
                device.get_energy()
                log.info(f'Energy consumption monitor available for device: {device}')
                self.readable_devices.append(device)
            except OSError:
                log.warning(f'Cannot read energy consumption from device {device}')

        self.energy_meter = EnergyMeter(self.readable_devices)

    def __enter__(self) -> EnergyMeter:
        self.energy_meter.start(self.init_tag)
        return self.energy_meter

    def __exit__(self, type_, value, traceback):
        self.energy_meter.stop()

        if self.enable_tracking and len(self.readable_devices) > 0:
            self.handler.process(self.energy_meter.get_trace())
            energy_trace_df = self.handler.get_dataframe()

            with TemporaryDirectory() as temp_dir:
                energy_trace_df.to_csv(f'{temp_dir}/df.csv', index=False)
                self.experiment['monitoring/energy'].upload(f'{temp_dir}/df.csv')
                self.experiment.sync()
        else:
            log.warning('No readable devices to publish energy consumption data')
