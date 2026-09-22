import time
import queue
import threading
import numpy as np

# Import AppConfig from daq_config.py
from lib_ipmu_daq_config import AppConfig


class Generator:
    """
    Generates mock signal data and puts it into a queue.
    Runs in its own thread.
    """
    def __init__(self, config: AppConfig, buf_q: queue.Queue, stop_event: threading.Event):
        """Initializes the Generator."""
        self.cfg = config
        self.buf_q = buf_q
        self.stop_event = stop_event

    def run(self):
        """
        The main loop for the generator thread.
        Continuously generates data until the stop event is set.
        """
        chunk_idx = 0
        next_t = time.perf_counter()
        
        gen_chunk_sec = self.cfg.io.gen_chunk_sec
        rel_axis_mock = self.cfg.dependent.rel_axis_mock

        while not self.stop_event.is_set():
            base = chunk_idx * gen_chunk_sec
            t_axis = rel_axis_mock + base
            
            pulse_A = self._genChunkPulse(t_axis, phase=self.cfg.debug_encoder.pulse_phase_A)
            pulse_B = self._genChunkPulse(t_axis, phase=self.cfg.dependent.pulse_phase_B)

            omega = self.cfg.dependent.omega
            Iu = self._genChunkSin(t_axis, A=0.3, omega=omega, phase=0.0)
            Iv = self._genChunkSin(t_axis, A=0.3, omega=omega, phase=2 * np.pi / 3)
            Iw = self._genChunkSin(t_axis, A=0.3, omega=omega, phase=4 * np.pi / 3)
            Vu = self._genChunkSin(t_axis, A=1.0, omega=omega, phase=np.pi / 8)
            Vv = self._genChunkSin(t_axis, A=1.0, omega=omega, phase=2 * np.pi / 3 + np.pi / 8)
            Vw = self._genChunkSin(t_axis, A=1.0, omega=omega, phase=4 * np.pi / 3 + np.pi / 8)

            try:
                self.buf_q.put_nowait((t_axis, pulse_A, pulse_B, Iu, Iv, Iw, Vu, Vv, Vw))
            except queue.Full:
                pass

            chunk_idx += 1
            next_t += gen_chunk_sec
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.perf_counter()
        
        print("Generator loop finished.")

    def _genChunkPulse(self, t: np.ndarray, phase: float) -> np.ndarray:
        """Generates a pulse wave chunk."""
        cfg = self.cfg.debug_encoder
        dep_cfg = self.cfg.dependent
        mod = (t + phase) % dep_cfg.pulse_width
        return np.where(mod < cfg.pulse_duty * dep_cfg.pulse_width, cfg.pulse_height, 0.0).astype(np.float32)

    def _genChunkSin(self, t: np.ndarray, A: float, omega: float, phase: float) -> np.ndarray:
        """Generates a sine wave chunk with noise."""
        noise = np.random.randn(len(t))
        return A * np.sin(omega * t + phase) + A * 0.01 * noise