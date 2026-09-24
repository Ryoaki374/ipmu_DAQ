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
        next_sample_index = 0

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
            sample_rate = task.timing.samp_clk_rate

            while not self.stop_event.is_set():
                data = np.asarray(task.read(number_of_samples_per_channel=n_samples_gen))
                sample_indices = np.arange(
                    next_sample_index,
                    next_sample_index + n_samples_gen,
                    dtype=np.int64,
                )
                t_ax = sample_indices.astype(np.float64) / sample_rate
                next_sample_index += n_samples_gen
                try:
                    self.buf_q.put_nowait((sample_indices, sample_rate, t_ax, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]))
                except queue.Full:
                    pass

    def _singleDataAcquisition(self, sample_rate):
        try:
            with nidaqmx.Task() as task:
                # Add analog input channels for current and voltage measurements
                task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai0")  # NI9215-0
                task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai1")  # NI9215-1
                # task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai2")  # NI9215-2
                # task.ai_channels.add_ai_voltage_chan("cDAQ2Mod1/ai3")  # NI9215-3
                # For Current
                #task.ai_channels.add_ai_voltage_chan("")
                #task.ai_channels.add_ai_voltage_chan("")
                #task.ai_channels.add_ai_voltage_chan("")
                ## For Voltage
                #task.ai_channels.add_ai_voltage_chan("")
                #task.ai_channels.add_ai_voltage_chan("")
                #task.ai_channels.add_ai_voltage_chan("")


                # Configure the sampling clock for continuous acquisition
                task.timing.cfg_samp_clk_timing(
                    rate=sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=int(sample_rate),
                )
                # Read a block of data (number of samples per channel equals sampling_rate)
                data = np.array(task.read(number_of_samples_per_channel=sample_rate))
                timedata = np.arange(sample_rate, dtype=np.float64) / sample_rate
                arr = np.vstack([timedata, data])
        except nidaqmx.errors.DaqError as e:
            print(f"Reading Error: {e}")
        return arr
