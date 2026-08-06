import time
import queue
import threading
import numpy as np
from itertools import islice

import nidaqmx
from nidaqmx.constants import AcquisitionType


class DataAquisition:
    def __init__(self, buf_q: queue.Queue, stop_event: threading.Event):
        """Initializes the DataAquisition."""
        self.buf_q = buf_q
        self.stop_event = stop_event

    def run(self):
        """
        The main loop for the DataAquisition thread.
        Continuously aquire data until the stop event is set.
        """

        sample_rate = 10000
        n_samples_gen = int(0.1 * sample_rate)

        tp = self._genTimeAxis(sample_rate)

        with nidaqmx.Task() as task:
            # For Current
            task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai0")
            task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai1")
            # For Voltage
            task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai2")
            task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai3")
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=n_samples_gen,
            )

            while not self.stop_event.is_set():
                data = np.asarray(
                    task.read(number_of_samples_per_channel=n_samples_gen)
                )
                t_ax = np.fromiter(
                    (next(tp) for _ in range(n_samples_gen)),
                    dtype=np.float32,
                    count=n_samples_gen,
                )
                try:
                    self.buf_q.put_nowait(
                        (
                            t_ax,
                            data[0],
                            data[1],
                            data[2],
                            data[3],
                        )
                    )
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

    def _singleDataAcquisition(self, sample_rate):
        try:
            tp = self._genTimeAxis(sample_rate)
            with nidaqmx.Task() as task:
                # Add analog input channels for current and voltage measurements
                task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai0")  # NI9215-0
                task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai1")  # NI9215-1
                # task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai2")  # NI9215-2
                # task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai3")  # NI9215-3

                # Configure the sampling clock for continuous acquisition
                task.timing.cfg_samp_clk_timing(
                    rate=sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=int(sample_rate),
                )
                # Read a block of data (number of samples per channel equals sampling_rate)
                data = np.array(task.read(number_of_samples_per_channel=sample_rate))
                timedata = np.array(list(islice(tp, sample_rate)))
                arr = np.vstack([timedata, data])
        except nidaqmx.errors.DaqError as e:
            print(f"Reading Error: {e}")
        return arr
