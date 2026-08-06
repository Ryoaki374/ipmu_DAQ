import time
import queue
import threading
import numpy as np

import lib_ipmu_recoder_config as config


class Generator:
    """
    Generates mock signal data and puts it into a queue.
    Runs in its own thread.
    """

    def __init__(self, buf_q: queue.Queue, stop_event: threading.Event):
        self.buf_q = buf_q
        self.stop_event = stop_event

    def run(self):
        """
        The main loop for the generator thread.
        Continuously generates data until the stop event is set.
        """
        chunk_idx = 0
        next_t = time.perf_counter()

        gen_chunk_sec = config.GEN_CHUNK_SEC

        while not self.stop_event.is_set():
            chunk = self._genChunk(chunk_idx)

            try:
                self.buf_q.put_nowait(chunk)
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

    def _genChunk(self, chunk_idx: int) -> tuple[np.ndarray, ...]:
        """Generate one chunk whose size follows the current sample rate."""
        sample_rate = config.SAMPLING_RATE
        samples_per_chunk = int(sample_rate * config.GEN_CHUNK_SEC)
        relative_time = (
            np.arange(samples_per_chunk, dtype=np.float64) / sample_rate
        )
        t_axis = relative_time + chunk_idx * config.GEN_CHUNK_SEC

        pulse_width = 1 / (
            config.MOCK_INPUT_VELOCITY * config.ENCODER_PULSES_PER_REVOLUTION
        )
        pulse_phase_B = -pulse_width / 4
        pulse_A = self._genChunkPulse(t_axis, phase=0.0)
        pulse_B = self._genChunkPulse(t_axis, phase=pulse_phase_B)
        pulse_C = pulse_A.copy()
        pulse_D = pulse_B.copy()
        return t_axis, pulse_A, pulse_B, pulse_C, pulse_D

    def _genChunkPulse(self, t: np.ndarray, phase: float) -> np.ndarray:
        """Generates a pulse wave chunk."""
        pulse_width = 1 / (
            config.MOCK_INPUT_VELOCITY * config.ENCODER_PULSES_PER_REVOLUTION
        )
        mod = (t + phase) % pulse_width
        return np.where(
            mod < 0.5 * pulse_width, config.PULSE_HEIGHT, 0.0
        ).astype(np.float32)
