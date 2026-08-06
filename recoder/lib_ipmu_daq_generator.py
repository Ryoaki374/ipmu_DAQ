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

            omega = 36
            Iu = self._genChunkSin(t_axis, A=0.3, omega=omega, phase=0.0)
            Iv = self._genChunkSin(t_axis, A=0.3, omega=omega, phase=2 * np.pi / 3)
            Vu = self._genChunkSin(t_axis, A=1.0, omega=omega, phase=np.pi / 8)
            Vv = self._genChunkSin(
                t_axis, A=1.0, omega=omega, phase=2 * np.pi / 3 + np.pi / 8
            )

            try:
                self.buf_q.put_nowait(
                    (
                        t_axis,
                        Iu,
                        Iv,
                        Vu,
                        Vv,
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

    def _genChunkSin(
        self, t: np.ndarray, A: float, omega: float, phase: float
    ) -> np.ndarray:
        """Generates a sine wave chunk with noise."""
        noise = np.random.randn(len(t))
        return A * np.sin(omega * t + phase) + A * 0.01 * noise
