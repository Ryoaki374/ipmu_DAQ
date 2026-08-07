import queue
import threading
import time
from collections import deque

import h5py
import numpy as np
import logging

import lib_ipmu_recoder_config as config


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
        proc_interval = config.PROCESS_INTERVAL
        samples_per_process = int(config.SAMPLING_RATE * proc_interval)
        next_proc = time.perf_counter() + proc_interval

        last_epoch = 0.0
        pruning = 1
        last_A = None
        last_B = None
        cum_count = 0

        # python ring buffer
        ring_t: deque[np.float32] = deque()
        ring_pulse_A: deque[np.float32] = deque()
        ring_pulse_B: deque[np.float32] = deque()

        while not self.stop_event.is_set():
            try:
                while True:
                    (
                        t,
                        pulse_A,
                        pulse_B,
                    ) = self.buf_q.get_nowait()
                    ring_t.extend(t)
                    ring_pulse_A.extend(pulse_A)
                    ring_pulse_B.extend(pulse_B)
                    self.buf_q.task_done()
            except queue.Empty:
                pass

            now = time.perf_counter()
            epoch = now - t0
            if now < next_proc:
                time.sleep(max(0, next_proc - now))
                continue

            next_proc += proc_interval

            if len(ring_t) < samples_per_process:
                continue

            # ----------Copy deque -> NumPy ----------
            samples_proc = samples_per_process
            t_blk = np.array(
                [ring_t.popleft() for _ in range(samples_proc)], dtype=np.float32
            )
            pulse_A_blk = np.array(
                [ring_pulse_A.popleft() for _ in range(samples_proc)],
                dtype=np.float32,
            )
            pulse_B_blk = np.array(
                [ring_pulse_B.popleft() for _ in range(samples_proc)],
                dtype=np.float32,
            )
            dir_log, last_A, last_B = self._getPulseDirection(
                pulse_A_blk,
                pulse_B_blk,
                threshold=config.ENCODER_THRESHOLD,
                prev_A=last_A,
                prev_B=last_B,
            )
            quad_sig = self._genQuadPulse(t_blk, dir_log)
            delta_cnt = self._getPulseCount(dir_log)
            cum_count += delta_cnt
            velocity = (
                delta_cnt
                / proc_interval
                / config.ENCODER_COUNTS_PER_REVOLUTION
            )

            try:
                self.quad_q.put_nowait(
                    (
                        t_blk,
                        pulse_A_blk,
                        pulse_B_blk,
                        quad_sig,
                        cum_count,
                        velocity,
                    )
                )
            except queue.Full:
                pass

            if self.DEBUG:
                # now = time.perf_counter()
                jitter = (epoch - last_epoch) * 1e3
                self.logger.info(
                    "epoch=%f, jitter=%f ms, count=%d, velocity=%f rps",
                    epoch,
                    jitter,
                    cum_count,
                    velocity,
                )
                last_epoch = epoch
            # ---------- append to HDF5 buffer ----------

            try:
                n = self.dset.shape[0]
                self.dset.resize(n + len(t_blk[::pruning]), axis=0)
                self.dset[n:] = np.array(
                    (
                        t_blk[::pruning],
                        pulse_A_blk[::pruning],
                        pulse_B_blk[::pruning],
                        quad_sig[::pruning],
                    )
                ).T
            except Exception as e:
                print(f"An error occurred during HDF5 write: {e}")

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
        samples = int(config.QUAD_PULSE_WIDTH * config.SAMPLING_RATE)
        if samples <= 0:
            return np.zeros_like(dir_log, dtype=np.float32)

        # This is equivalent to convolution with a rectangular pulse, but is
        # O(N) instead of O(N * samples).  The difference is substantial at
        # 1 MHz, where the pulse width spans 250 samples.
        cumulative = np.cumsum(dir_log, dtype=np.float32)
        quad_sig = cumulative.copy()
        if samples < len(quad_sig):
            quad_sig[samples:] -= cumulative[:-samples]
        quad_sig *= config.PULSE_HEIGHT
        return quad_sig

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
