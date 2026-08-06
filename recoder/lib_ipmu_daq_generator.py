import time
import queue
import threading
import numpy as np


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

        gen_chunk_sec = 0.2
        n_samples_gen = int(10000 * gen_chunk_sec)
        rel_axis_mock = rel_axis_mock = (
            np.arange(n_samples_gen, dtype=np.float32) / 1000
        )

        while not self.stop_event.is_set():
            base = chunk_idx * gen_chunk_sec
            t_axis = rel_axis_mock + base

            pulse_width = 1 / (1 * 512)  # [s]
            pulse_phase_B = -pulse_width / 4
            pulse_A = self._genChunkPulse(t_axis, phase=0.0)
            pulse_B = self._genChunkPulse(t_axis, phase=pulse_phase_B)
            pulse_C = self._genChunkPulse(t_axis, phase=0.0)
            pulse_D = self._genChunkPulse(t_axis, phase=pulse_phase_B)

            try:
                self.buf_q.put_nowait(
                    (
                        t_axis,
                        pulse_A,
                        pulse_B,
                        pulse_C,
                        pulse_D,
                    )
                )
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
        pulse_width = 1 / (1 * 512)  # [s]
        mod = (t + phase) % pulse_width
        return np.where(mod < 0.5 * pulse_width, 5, 0.0).astype(np.float32)
