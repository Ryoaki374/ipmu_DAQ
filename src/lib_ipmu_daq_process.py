import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

# Import AppConfig from daq_config.py
from lib_ipmu_daq_config import AppConfig

class Processor:
    """
    Processes raw signal data, logs it, and sends results to the GUI queue.
    Runs in its own thread.
    """
    def __init__(self, config: AppConfig, buf_q: queue.Queue, quad_q: queue.Queue, stop_event: threading.Event, runs_dir: Path):
        """Initializes the Processor."""
        self.cfg = config
        self.buf_q = buf_q
        self.quad_q = quad_q
        self.stop_event = stop_event
        self.runs_dir = runs_dir
        self.h5f = None
        self.dset = None

    def run(self):
        """
        The main loop for the processor thread.
        Initializes storage, then processes data until the stop event is set.
        """
        self._initStorer()
        self._processorLoop()

    def _initStorer(self):
        """Initializes settings for saving data to an HDF5 file."""
        self.runs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        filepath = self.runs_dir / f"{timestamp}.h5"
        
        self.h5f = h5py.File(filepath, "w")
        self.dset = self.h5f.create_dataset(
            "log",
            shape=(0, self.cfg.logging.log_data_num),
            maxshape=(None, self.cfg.logging.log_data_num),
            dtype=np.float32,
            chunks=(self.cfg.logging.log_chunk, self.cfg.logging.log_data_num),
            compression="gzip"
        )
        print(f"HDF5 dataset created at: {filepath}")

    def _processorLoop(self):
        """Processes raw data, saves logs, and sends results to the GUI queue."""
        ideal_provider = self._getCommandVelo()
        buf = np.empty((self.cfg.logging.log_chunk, self.cfg.logging.log_data_num), dtype=np.float32)
        buf_len = 0
        
        ring_t: deque[np.float32] = deque(); ring_a: deque[np.float32] = deque(); ring_b: deque[np.float32] = deque()
        ring_Iu: deque[np.float32] = deque(); ring_Iv: deque[np.float32] = deque(); ring_Iw: deque[np.float32] = deque()
        ring_Vu: deque[np.float32] = deque(); ring_Vv: deque[np.float32] = deque(); ring_Vw: deque[np.float32] = deque()

        last_A = last_B = None
        next_proc = time.perf_counter()
        cum_count = 0

        while not self.stop_event.is_set():
            try:
                while True:
                    t, pA, pB, Iu, Iv, Iw, Vu, Vv, Vw = self.buf_q.get_nowait()
                    ring_t.extend(t); ring_a.extend(pA); ring_b.extend(pB)
                    ring_Iu.extend(Iu); ring_Iv.extend(Iv); ring_Iw.extend(Iw)
                    ring_Vu.extend(Vu); ring_Vv.extend(Vv); ring_Vw.extend(Vw)
                    self.buf_q.task_done()
            except queue.Empty:
                pass

            now = time.perf_counter()
            if now < next_proc:
                time.sleep(max(0, next_proc - now))
                continue
            next_proc += self.cfg.io.proc_interval

            if len(ring_t) < self.cfg.dependent.samples_proc:
                continue

            samples_proc = self.cfg.dependent.samples_proc
            t_blk = np.array([ring_t.popleft() for _ in range(samples_proc)], dtype=np.float32)
            a_blk = np.array([ring_a.popleft() for _ in range(samples_proc)], dtype=np.float32)
            b_blk = np.array([ring_b.popleft() for _ in range(samples_proc)], dtype=np.float32)
            Iu_blk = np.array([ring_Iu.popleft() for _ in range(samples_proc)], dtype=np.float32)
            Iv_blk = np.array([ring_Iv.popleft() for _ in range(samples_proc)], dtype=np.float32)
            Iw_blk = np.array([ring_Iw.popleft() for _ in range(samples_proc)], dtype=np.float32)
            Vu_blk = np.array([ring_Vu.popleft() for _ in range(samples_proc)], dtype=np.float32)
            Vv_blk = np.array([ring_Vv.popleft() for _ in range(samples_proc)], dtype=np.float32)
            Vw_blk = np.array([ring_Vw.popleft() for _ in range(samples_proc)], dtype=np.float32)

            dir_log, last_A, last_B = self._getPulseDirection(a_blk, b_blk, threshold=self.cfg.encoder_postproc.threshold, prev_A=last_A, prev_B=last_B)
            quad_sig = self._genQuadPulse(t_blk, dir_log)
            delta_cnt = self._getPulseCount(dir_log)
            cum_count += delta_cnt
            velocity = delta_cnt / self.cfg.io.proc_interval / 2048

            time_p, P_u, P_v, P_w, P_tot = self._getPower(t_blk, Iu_blk, Iv_blk, Iw_blk, Vu_blk, Vv_blk, Vw_blk, 0, -0.1)
            t_ref, v_ref = ideal_provider(t_blk[0], t_blk[-1])

            try:
                self.quad_q.put_nowait((t_blk, a_blk, b_blk, quad_sig, t_blk[-1], cum_count, velocity, t_ref, v_ref, time_p, P_tot, Iu_blk, Vu_blk))
            except queue.Full:
                pass

            buf[buf_len] = (t_blk[-1], (v_ref[-1] if v_ref.size else 0.0), velocity, P_tot)
            buf_len += 1
            if buf_len == buf.shape[0]:
                n = self.dset.shape[0]
                self.dset.resize(n + buf_len, axis=0)
                self.dset[-buf_len:] = buf
                buf_len = 0
        
        if buf_len:
            n = self.dset.shape[0]
            self.dset.resize(n + buf_len, axis=0)
            self.dset[-buf_len:] = buf[:buf_len]
        
        if self.h5f:
            self.h5f.close()
            print("HDF5 file closed by processor.")
        print("Processor loop finished.")

    def _getPulseDirection(self, dA: np.ndarray, dB: np.ndarray, *, threshold: float, prev_A: bool | None = None, prev_B: bool | None = None) -> tuple[np.ndarray, bool, bool]:
        A = dA > threshold; B = dB > threshold
        A_prev = np.concatenate(([prev_A if prev_A is not None else A[0]], A[:-1]))
        B_prev = np.concatenate(([prev_B if prev_B is not None else B[0]], B[:-1]))
        dir_log = (B_prev ^ A).astype(int) - (A_prev ^ B).astype(int)
        return dir_log.astype(np.int8), bool(A[-1]), bool(B[-1])

    def _getPulseCount(self, dir_log: np.ndarray) -> int:
        return np.sum(dir_log)

    def _genQuadPulse(self, t: np.ndarray, dir_log: np.ndarray) -> np.ndarray:
        width = self.cfg.encoder_postproc.quadpulse_width
        height = self.cfg.debug_encoder.pulse_height
        sampling_rate = self.cfg.io.sample_rate
        samples = int(width * sampling_rate)
        if samples <= 0: return np.zeros_like(dir_log, dtype=np.float32)
        base = np.full(samples, height, dtype=np.float32)
        return np.convolve(dir_log, base, mode="full")[: len(t)]

    def _genCommandFreq(self):
        drv, dep, io = self.cfg.driver, self.cfg.dependent, self.cfg.io
        num_step = np.ceil((dep.num_pulses - drv.pps) * drv.target_freq / drv.step)
        time_arr = np.arange(0, drv.t_DC + num_step * drv.rst + dep.stable_sec, io.proc_interval, dtype=np.float32)
        freq = np.zeros_like(time_arr)
        n_DC = int(round(drv.t_DC / io.proc_interval))
        n_rst = int(round(drv.rst / io.proc_interval))
        for i in range(int(num_step)):
            start, end = n_DC + i * n_rst, n_DC + (i + 1) * n_rst
            freq[start:end] = drv.pps / dep.num_pulses + drv.target_freq / dep.num_pulses * drv.step * (i + 1)
        freq[int(n_DC + n_rst * num_step):] = drv.target_freq
        return time_arr, freq

    def _utilSchmittTrigger(self, upper, lower, current):
        y_schmitt = np.zeros_like(current)
        state = 0.0
        for i, sample in enumerate(current):
            if state == 0.0 and sample >= upper: state = 1.0
            elif state == 1.0 and sample <= lower: state = 0.0
            y_schmitt[i] = state
        d = np.diff(y_schmitt.astype(int))
        return np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1
    
    def _getPower(self, time_arr, I_u, I_v, I_w, V_u, V_v, V_w, upper, lower):
        rise_idx_I_u, _ = self._utilSchmittTrigger(upper, lower, I_u)
        if len(rise_idx_I_u) >= 2:
            s, e = rise_idx_I_u[0], rise_idx_I_u[1]
            P_u = np.mean(I_u[s:e] * V_u[s:e])
            P_v = np.mean(I_v[s:e] * V_v[s:e])
            P_w = np.mean(I_w[s:e] * V_w[s:e])
        else:
            P_u, P_v, P_w = np.mean(I_u * V_u), np.mean(I_v * V_v), np.mean(I_w * V_w)
        P_tot = P_u + P_v + P_w
        return time_arr.mean(), P_u, P_v, P_w, P_tot

    def _getCommandVelo(self):
        time_axis, velocity = self._genCommandFreq()
        def slicer(t_start: float, t_end: float):
            i0 = np.searchsorted(time_axis, t_start, side="left")
            i1 = np.searchsorted(time_axis, t_end, side="left")
            return time_axis[i0:i1], velocity[i0:i1]
        return slicer




