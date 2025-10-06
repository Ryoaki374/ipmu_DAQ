import time
import queue
import threading
import numpy as np

import nidaqmx
from nidaqmx.constants import AcquisitionType

# Import AppConfig from daq_config.py
from lib_ipmu_daq_config import AppConfig


class DataAquisition:
    def __init__(self, config: AppConfig, buf_q: queue.Queue, stop_event: threading.Event):
        """Initializes the DataAquisition."""
        self.cfg = config
        self.buf_q = buf_q
        self.stop_event = stop_event

    def run(self):
        """
        The main loop for the DataAquisition thread.
        Continuously aquire data until the stop event is set.
        """

        sample_rate = self.cfg.io.sample_rate
        n_samples_gen = self.cfg.dependent.n_samples_gen

        tp = self._genTimeAxis(sample_rate)

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai0")
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai1")
            # For Current
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod2/ai0")
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod2/ai1")
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod2/ai2")
            # For Voltage
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod3/ai0")
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod3/ai1")
            task.ai_channels.add_ai_voltage_chan("cDAQ2Mod3/ai2")
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=n_samples_gen,
            )

            while not self.stop_event.is_set():
                data = np.asarray(task.read(number_of_samples_per_channel=n_samples_gen))
                t_ax = np.fromiter(
                    (next(tp) for _ in range(n_samples_gen)),
                    dtype=np.float32,
                    count=n_samples_gen,
                )
                try:
                    self.buf_q.put_nowait((t_ax, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]))
                except queue.Full:
                    pass

    def _genTimeAxis(self, sample_rate):
        interval = 1.0 / sample_rate  # Time interval between samples
        start_time = time.perf_counter()
        next_sample_time = start_time
        while True:
            current_time = time.perf_counter()
            # Wait until the next scheduled sample time
            if current_time < next_sample_time:
                time.sleep(next_sample_time - current_time)

            # Yield the relative time since the start
            yield next_sample_time - start_time
            next_sample_time += interval