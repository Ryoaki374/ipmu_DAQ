import queue
import threading
import time
from collections import deque

import h5py
import numpy as np
import logging


class Processor:
    """
    Processes raw signal data, logs it, and sends results to the GUI queue.
    Runs in its own thread.
    """

    def __init__(
        self,
        buf_q: queue.Queue,
        quad_q: queue.Queue,
        comvel_q: queue.Queue,
        h5f: h5py.File,
        dset: h5py.Dataset,
        stop_event: threading.Event,
        logger: logging.Logger,
        debug: bool = False,
    ):
        self.buf_q = buf_q
        self.quad_q = quad_q
        self.comvel_q = comvel_q
        self.h5f = h5f
        self.dset = dset
        self.stop_event = stop_event
        self.DEBUG = debug
        self.logger = logger

    def run(self):
        """
        The main loop for the processor thread.
        Initializes storage, then processes data until the stop event is set.
        """
        self._processorLoop()

    def _processorLoop(self):
        """Processes raw data, saves logs, and sends results to the GUI queue."""
        t0 = time.perf_counter()
        proc_interval = 0.2
        next_proc = time.perf_counter() + proc_interval

        buf_hdf5_idx = 0
        last_epoch = 0.0
        pruning = 1
        last_A = None
        last_B = None
        t_ref = np.array([0.])
        v_ref = np.array([0.])

        # python ring buffer
        ring_t: deque[np.float32] = deque()
        ring_Iu: deque[np.float32] = deque()
        ring_Iv: deque[np.float32] = deque()
        ring_Vu: deque[np.float32] = deque()
        ring_Vv: deque[np.float32] = deque()

        while not self.stop_event.is_set():
            try:
                while True:
                    (
                        t,
                        Iu,
                        Iv,
                        Vu,
                        Vv,
                    ) = self.buf_q.get_nowait()
                    ring_t.extend(t)
                    ring_Iu.extend(Iu)
                    ring_Iv.extend(Iv)
                    ring_Vu.extend(Vu)
                    ring_Vv.extend(Vv)
                    self.buf_q.task_done()
            except queue.Empty:
                pass

            now = time.perf_counter()
            epoch = now - t0
            if now < next_proc:
                time.sleep(max(0, next_proc - now))
                continue

            if len(ring_t) < interval * 10000:
                continue

            # ----------Copy deque -> NumPy ----------
            samples_proc = int(interval * 10000)
            t_blk = np.array(
                [ring_t.popleft() for _ in range(samples_proc)], dtype=np.float32
            )
            Iu_blk = np.array(
                [ring_Iu.popleft() for _ in range(samples_proc)], dtype=np.float32
            )
            Iv_blk = np.array(
                [ring_Iv.popleft() for _ in range(samples_proc)], dtype=np.float32
            )
            Vu_blk = np.array(
                [ring_Vu.popleft() for _ in range(samples_proc)], dtype=np.float32
            )
            Vv_blk = np.array([ring_Vv.popleft() for _ in range(samples_proc)], dtype=np.float32)

            dir_log, last_A, last_B = self._getPulseDirection(a_blk, b_blk, threshold=self.cfg.encoder_postproc.threshold, prev_A=last_A, prev_B=last_B)
            quad_sig = self._genQuadPulse(t_blk, dir_log)
            delta_cnt = self._getPulseCount(dir_log)
            cum_count += delta_cnt
            velocity = delta_cnt / proc_interval / 2048
            vel_blk = np.full(len(t_blk[::pruning]), velocity)

            if self.DEBUG:
                # now = time.perf_counter()
                jitter = (epoch - last_epoch) * 1e3
                self.logger.info(
                    epoch,
                    jitter,
                )
                last_epoch = epoch
            # ---------- append to HDF5 buffer ----------

            try:
                n = self.dset.shape[0]
                self.dset.resize(n + len(t_blk[::pruning]), axis=0)
                self.dset[n:] = np.array(
                    (
                        t_blk[::pruning],
                        Iu_blk[::pruning],
                        Vu_blk[::pruning],
                        Iv_blk[::pruning],
                        Vv_blk[::pruning],
                    )
                ).T
            except Exception as e:
                print(f"An error occurred during HDF5 write: {e}")

        # --- Final flush less than 1024 data ---
        # if buf_hdf5_idx != 0:
        n = self.dset.shape[0]
        self.dset.resize(n + len(t_blk[::pruning]), axis=0)
        self.dset[n:] = np.array(
            (
                t_blk[::pruning],
                Iu_blk[::pruning],
                Vu_blk[::pruning],
                Iv_blk[::pruning],
                Vv_blk[::pruning],
            )
        ).T

        if self.h5f:
            self.h5f.close()
            print("HDF5 file closed by processor.")
        print("Processor loop finished.")

    def _utilSchmittTrigger(self, upper, lower, current):
        y_schmitt = np.zeros_like(current)
        state = 0.0
        for i, sample in enumerate(current):
            if state == 0.0 and sample >= upper:
                state = 1.0
            elif state == 1.0 and sample <= lower:
                state = 0.0
            y_schmitt[i] = state
        d = np.diff(
            y_schmitt.astype(int)
        )  # diff（+1 なら立ち上がり, −1 なら立ち下がり）
        rise_idx = np.where(d == 1)[0] + 1  # 立ち上がり位置
        fall_idx = np.where(d == -1)[0] + 1  # 立ち下がり位置
        return rise_idx, fall_idx

    def _getPulseDirection(
        self,
        dA: np.ndarray,
        dB: np.ndarray,
        *,
        threshold: float,
        prev_A: bool | None = None,
        prev_B: bool | None = None,
    ) -> tuple[np.ndarray, bool, bool]:
        A = dA > threshold
        B = dB > threshold
        if prev_A is None:  # first block -> old behaviour
            A_prev = np.concatenate(([A[0]], A[:-1]))
            B_prev = np.concatenate(([B[0]], B[:-1]))
        else:  # use states carried over from last block
            A_prev = np.concatenate(([prev_A], A[:-1]))
            B_prev = np.concatenate(([prev_B], B[:-1]))
        dir_log = (B_prev ^ A).astype(int) - (A_prev ^ B).astype(int)
        return dir_log.astype(np.int8), bool(A[-1]), bool(B[-1])

    def _getPulseCount(self, dir_log: np.ndarray) -> int:
        return np.sum(dir_log)

    def _genQuadPulse(self, t: np.ndarray, dir_log: np.ndarray) -> np.ndarray:
        width = self.cfg.encoder_postproc.quadpulse_width
        height = self.cfg.debug_encoder.pulse_height
        sampling_rate = self.cfg.io.sample_rate
        samples = int(width * sampling_rate)
        if samples <= 0:
            return np.zeros_like(dir_log, dtype=np.float32)
        base = np.full(samples, height, dtype=np.float32)
        return np.convolve(dir_log, base, mode="full")[: len(t)]

    def _addNewDatasetToHDF(self, current: int):
        reduction_group = self.h5f["current_reduction"]
        dataset_name = f"current_{current}"

        if dataset_name in reduction_group:
            print(f"Dataset '{dataset_name}' already exists.")
            return reduction_group[dataset_name]

        new_dset = reduction_group.create_dataset(
            dataset_name,
            shape=(0, self.cfg.logging.log_data_num),
            maxshape=(None, self.cfg.logging.log_data_num),
            dtype=np.float32,
            compression="gzip",
        )
        print(f"Created new dataset: {dataset_name}")
        return new_dset
