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
        interval = 0.1
        next_proc = time.perf_counter() + interval

        buf_hdf5_idx = 0
        last_epoch = 0.0
        pruning = 1

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
            Vv_blk = np.array(
                [ring_Vv.popleft() for _ in range(samples_proc)], dtype=np.float32
            )

            # get power
            # idx, time_p, P_u, P_v, P_tot_sum = self._getPower(
            #    t_blk, Iu_blk, Iv_blk, Vu_blk, Vv_blk, 0.1, -0.1
            # )
            # extend power for instantaneous power array
            # P_u_blk = np.full(len(t_blk[::pruning]), P_u)
            # P_v_blk = np.full(len(t_blk[::pruning]), P_v)
            # P_tot_sum_blk = np.full(len(t_blk[::pruning]), P_tot_sum)

            # get squared current
            # _I2u, _I2v = self._getSquaredCurrent(Iu_blk, Iv_blk, idx)

            # _I2u_blk = np.full(len(t_blk[::pruning]), _I2u)
            # _I2v_blk = np.full(len(t_blk[::pruning]), _I2v)

            if self.DEBUG:
                # now = time.perf_counter()
                jitter = (epoch - last_epoch) * 1e3
                self.logger.info(
                    # "EPOCH = %f, jitter = %6.2f ms, time_p = %f, P_tot = %f, _Iu = %f, _Iv = %f",
                    "EPOCH = %f, jitter = %6.2f ms",
                    epoch,
                    jitter,
                    # time_p,
                    # P_tot_sum,
                    # _I2u,
                    # _I2v,
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
                        # P_tot_sum_blk,
                        # _I2u_blk,
                        # _I2v_blk,
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
                # P_tot_sum_blk,
                # _I2u_blk,
                # _I2v_blk,
            )
        ).T

        if self.h5f:
            self.h5f.close()
            print("HDF5 file closed by processor.")
        print("Processor loop finished.")

    def _processorLoop(self):
        """Processes raw data, saves logs, and sends results to the GUI queue."""
        t0 = time.perf_counter()
        interval = 0.1
        next_proc = time.perf_counter() + interval

        buf_hdf5_idx = 0
        last_epoch = 0.0
        pruning = 1

        # python ring buffer
        ring_t: deque[np.float32] = deque()
        ring_Iu: deque[np.float32] = deque()
        ring_Iv: deque[np.float32] = deque()

        while not self.stop_event.is_set():
            try:
                while True:
                    (
                        t,
                        Iu,
                        Iv,
                    ) = self.buf_q.get_nowait()
                    ring_t.extend(t)
                    ring_Iu.extend(Iu)
                    ring_Iv.extend(Iv)
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

            if self.DEBUG:
                # now = time.perf_counter()
                jitter = (epoch - last_epoch) * 1e3
                self.logger.info(
                    # "EPOCH = %f, jitter = %6.2f ms, time_p = %f, P_tot = %f, _Iu = %f, _Iv = %f",
                    "EPOCH = %f, jitter = %6.2f ms",
                    epoch,
                    jitter,
                    # time_p,
                    # P_tot_sum,
                    # _I2u,
                    # _I2v,
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
                        Iv_blk[::pruning],
                        # P_tot_sum_blk,
                        # _I2u_blk,
                        # _I2v_blk,
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
                Iv_blk[::pruning],
                # P_tot_sum_blk,
                # _I2u_blk,
                # _I2v_blk,
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

    # revised version (2025/11/17)
    def _getPower(self, time_arr, I_u, I_v, V_u, V_v, upper, lower):
        rise_idx_I_u, _ = self._utilSchmittTrigger(upper, lower, I_u)

        if len(rise_idx_I_u) >= 2:
            s, f = rise_idx_I_u[0], rise_idx_I_u[1]
            P_u = np.mean(I_u[s:f] * V_u[s:f])
            P_v = np.mean(I_v[s:f] * V_v[s:f])
        else:
            (
                P_u,
                P_v,
            ) = (
                np.mean(I_u * V_u),
                np.mean(I_v * V_v),
            )
        P_tot = P_u + P_v
        return rise_idx_I_u, time_arr.mean(), P_u, P_v, P_tot

    def _getSquaredCurrent(self, I_u, I_v, idx):
        # rise_idx_I_u, _ = self._utilSchmittTrigger(upper, lower, I_u)
        rise_idx_I_u = idx
        if len(rise_idx_I_u) >= 2:
            s, f = rise_idx_I_u[0], rise_idx_I_u[1]
            _I_u = np.mean(I_u[s:f] * I_u[s:f])
            _I_v = np.mean(I_v[s:f] * I_v[s:f])
        else:
            _I_u, _I_v = (
                np.mean(I_u * I_u),
                np.mean(I_v * I_v),
            )
        return _I_u, _I_v
