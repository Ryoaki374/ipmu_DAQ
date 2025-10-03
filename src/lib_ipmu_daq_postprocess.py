import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

# Import AppConfig from daq_config.py
from lib_ipmu_daq_config import AppConfig


class DAQApp:
    """
    Main application class for data acquisition, processing, and display.
    This class is designed to prepare the DAQ environment and provide
    controllable thread objects to a higher-level application.
    """

    def __init__(self, config: AppConfig, runs_dir: str | Path):
        """
        Initializes the DAQApp with configuration.
        Does not perform heavy setup here. Call setup() for that.
        """
        self.cfg = config
        self.runs_dir = Path(runs_dir)
        self.is_setup = False
        print("DAQApp instance created. Call setup() to prepare for running.")

    def setup(self):
        """
        Sets up all necessary components for the application.
        This method should be called before creating threads.
        """
        if self.is_setup:
            print("Setup has already been called.")
            return

        print("Setting up DAQApp components...")
        self._initState()
        self._initBuffers()
        self._initThreads()
        self._initStorer()
        self.is_setup = True
        print("Setup complete.")


    def _initState(self):
        """Initializes the application state."""
        self.DEBUG = True  # Enable debug mode for now
        self.ideal_provider = None


    def _initBuffers(self):
        """Initializes queues for data exchange."""
        self.buf_q = queue.Queue(maxsize=self.cfg.io.queue_depth)  # raw data
        self.quad_q = queue.Queue(maxsize=self.cfg.io.quad_depth) # processed data


    def _initThreads(self):
        """Initializes the event for stopping threads."""
        self.stop_event = threading.Event()


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
        print(f"dset shape    = {self.dset.shape}")
        print(f"dset maxshape = {self.dset.maxshape}")


    def createGeneratorThread(self) -> threading.Thread:
        """
        Creates and returns the generator thread object.
        The caller is responsible for starting and joining this thread.
        """
        if not self.is_setup:
            raise RuntimeError("You must call setup() before creating threads.")
        generator_thread = threading.Thread(target=self._generatorLoop, daemon=True)
        return generator_thread

    def createProcessorThread(self) -> threading.Thread:
        """
        Creates and returns the processor thread object.
        The caller is responsible for starting and joining this thread.
        """
        if not self.is_setup:
            raise RuntimeError("You must call setup() before creating threads.")
        processor_thread = threading.Thread(target=self._processorLoop, daemon=True)
        return processor_thread
    
    def signalStop(self):
        """
        Signals the threads to stop their loops.
        """
        print("Signaling threads to stop...")
        self.stop_event.set()

    # ------------------------------------------------------------
    # Core Logic Methods (Producer & Consumer)
    # ------------------------------------------------------------

    def _generatorLoop(self):
        """
        Producer thread loop. Generates mock signals and puts them into a queue.
        """
        chunk_idx = 0
        next_t = time.perf_counter()
        
        # Access config values via self.cfg
        gen_chunk_sec = self.cfg.io.gen_chunk_sec
        rel_axis_mock = self.cfg.dependent.rel_axis_mock

        while not self.stop_event.is_set():
            base = chunk_idx * gen_chunk_sec
            t_axis = rel_axis_mock + base
            
            # --- Generate mock encoder signals ---
            pulse_A = self._genChunkPulse(t_axis, phase=self.cfg.debug_encoder.pulse_phase_A)
            pulse_B = self._genChunkPulse(t_axis, phase=self.cfg.dependent.pulse_phase_B)

            # --- Generate mock power signals ---
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
                pass # Or log a warning

            chunk_idx += 1
            next_t += gen_chunk_sec
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # To prevent drift if processing takes too long
                next_t = time.perf_counter()
        
        print("Generator loop finished.")

    def _processorLoop(self):
        """
        Consumer thread loop. Processes raw data, saves logs, and sends results to the GUI queue.
        """
        # ---------- one-time init ----------
        self.ideal_provider = self._getCommandVelo()

        # ---------- logging buffer ----------
        buf = np.empty((self.cfg.logging.log_chunk, self.cfg.logging.log_data_num), dtype=np.float32)
        buf_len = 0
        print("Processor buffer shape =", buf.shape)

        # --- Ring buffers using Python deque ---
        ring_t: deque[np.float32] = deque(); ring_a: deque[np.float32] = deque(); ring_b: deque[np.float32] = deque()
        ring_Iu: deque[np.float32] = deque(); ring_Iv: deque[np.float32] = deque(); ring_Iw: deque[np.float32] = deque()
        ring_Vu: deque[np.float32] = deque(); ring_Vv: deque[np.float32] = deque(); ring_Vw: deque[np.float32] = deque()

        last_A = last_B = None
        next_proc = time.perf_counter()
        cum_count = 0

        while not self.stop_event.is_set():
            # ---------- Fill buffers with non-blocking reads ----------
            try:
                while True:
                    t, pA, pB, Iu, Iv, Iw, Vu, Vv, Vw = self.buf_q.get_nowait()
                    ring_t.extend(t); ring_a.extend(pA); ring_b.extend(pB)
                    ring_Iu.extend(Iu); ring_Iv.extend(Iv); ring_Iw.extend(Iw)
                    ring_Vu.extend(Vu); ring_Vv.extend(Vv); ring_Vw.extend(Vw)
                    self.buf_q.task_done()
            except queue.Empty:
                pass

            # ---------- Wait until the next processing time ----------
            now = time.perf_counter()
            if now < next_proc:
                time.sleep(next_proc - now)
                continue
            next_proc += self.cfg.io.proc_interval

            # ---------- Skip if not enough samples ----------
            if len(ring_t) < self.cfg.dependent.samples_proc:
                continue

            # ---------- Copy from deque to NumPy for processing ----------
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

            # ---------- Signal Processing ----------
            dir_log, last_A, last_B = self._getPulseDirection(
                a_blk, b_blk,
                threshold=self.cfg.encoder_postproc.threshold,
                prev_A=last_A, prev_B=last_B
            )
            quad_sig = self._genQuadPulse(t_blk, dir_log)
            delta_cnt = self._getPulseCount(dir_log)
            cum_count += delta_cnt
            velocity = delta_cnt / self.cfg.io.proc_interval / 2048 # 2048 is specific constant

            # ---------- Power calculation ----------
            time_p, P_u, P_v, P_w, P_tot = self._getPower(
                t_blk, Iu_blk, Iv_blk, Iw_blk,
                Vu_blk, Vv_blk, Vw_blk, 0, -0.1
            )
            
            # ---------- Slice ideal velocity ----------
            t_ref, v_ref = self.ideal_provider(t_blk[0], t_blk[-1])

            # ---------- Send to GUI queue ----------
            try:
                self.quad_q.put_nowait(
                    (t_blk, a_blk, b_blk, quad_sig,
                     t_blk[-1], cum_count, velocity,
                     t_ref, v_ref, time_p, P_tot, Iu_blk, Vu_blk)
                )
            except queue.Full:
                pass

            # ---- Append to HDF5 buffer and flush if full ----
            buf[buf_len] = (t_blk[-1], (v_ref[-1] if v_ref.size else 0.0), velocity, P_tot)
            buf_len += 1
            if buf_len == buf.shape[0]:
                n = self.dset.shape[0]
                self.dset.resize(n + buf_len, axis=0)
                self.dset[-buf_len:] = buf
                buf_len = 0

        # ---- Final flush of HDF5 buffer ----
        if buf_len:
            n = self.dset.shape[0]
            self.dset.resize(n + buf_len, axis=0)
            self.dset[-buf_len:] = buf[:buf_len]
        
        if self.h5f:
            self.h5f.close()
            print("HDF5 file closed by processor.")
        print("Processor loop finished.")


    # ------------------------------------------------------------
    # Helper methods (ported from original script)
    # ------------------------------------------------------------

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

    def _getPulseDirection(self, dA: np.ndarray, dB: np.ndarray, *, threshold: float, 
                           prev_A: bool | None = None, prev_B: bool | None = None
    ) -> tuple[np.ndarray, bool, bool]:
        """Determines the direction of pulse movement from two channels."""
        # --- current logic level ---
        A = dA > threshold
        B = dB > threshold

        # --- previous sample for XOR ---
        if prev_A is None:  # first block
            A_prev = np.concatenate(([A[0]], A[:-1]))
            B_prev = np.concatenate(([B[0]], B[:-1]))
        else:  # use states carried over from last block
            A_prev = np.concatenate(([prev_A], A[:-1]))
            B_prev = np.concatenate(([prev_B], B[:-1]))

        dir_log = (B_prev ^ A).astype(int) - (A_prev ^ B).astype(int)
        return dir_log.astype(np.int8), bool(A[-1]), bool(B[-1])

    def _getPulseCount(self, dir_log: np.ndarray) -> int:
        """Counts the net number of pulses."""
        return np.sum(dir_log)

    def _genQuadPulse(self, t: np.ndarray, dir_log: np.ndarray) -> np.ndarray:
        """Generates a quadrature pulse signal for visualization."""
        width = self.cfg.encoder_postproc.quadpulse_width
        height = self.cfg.debug_encoder.pulse_height
        sampling_rate = self.cfg.io.sample_rate
        
        samples = int(width * sampling_rate)
        if samples <= 0:
            return np.zeros_like(dir_log, dtype=np.float32)
        base = np.full(samples, height, dtype=np.float32)
        return np.convolve(dir_log, base, mode="full")[: len(t)]

    def _genCommandFreq(self):
        """Generates the time and frequency arrays for the ideal command velocity."""
        drv = self.cfg.driver
        dep = self.cfg.dependent
        io = self.cfg.io
        
        num_step = np.ceil((dep.num_pulses - drv.pps) * drv.target_freq / drv.step)
        time_arr = np.arange(0, drv.t_DC + num_step * drv.rst + dep.stable_sec, io.proc_interval, dtype=np.float32)
        freq = np.zeros_like(time_arr, dtype=np.float32)
        
        n_DC = int(np.round(drv.t_DC / io.proc_interval))
        freq[:n_DC] = 0.0
        n_rst = int(np.round(drv.rst / io.proc_interval))
        
        for i in range(int(num_step)):
            start_idx = n_DC + i * n_rst
            end_idx = n_DC + (i + 1) * n_rst
            freq[start_idx:end_idx] = drv.pps / dep.num_pulses + drv.target_freq / dep.num_pulses * drv.step * (i + 1)
        
        freq[int(n_DC + n_rst * num_step):] = drv.target_freq
        return time_arr, freq

    def _utilSchmittTrigger(self, upper, lower, current):
        """A simple Schmitt trigger implementation."""
        y_schmitt = np.zeros_like(current)
        state = 0.0
        for i, sample in enumerate(current):
            if state == 0.0 and sample >= upper:
                state = 1.0
            elif state == 1.0 and sample <= lower:
                state = 0.0
            y_schmitt[i] = state

        d = np.diff(y_schmitt.astype(int))
        rise_idx = np.where(d == 1)[0] + 1
        fall_idx = np.where(d == -1)[0] + 1
        return rise_idx, fall_idx
    
    def _getPower(self, time_arr, I_u, I_v, I_w, V_u, V_v, V_w, upper, lower):
        """Calculates the electrical power from current and voltage signals."""
        rise_idx_I_u, _ = self._utilSchmittTrigger(upper, lower, I_u)
        
        # If more than one full period is available, calculate power over one period
        if len(rise_idx_I_u) >= 2:
            period = rise_idx_I_u[1] - rise_idx_I_u[0]
            P_u = np.mean(I_u[rise_idx_I_u[0]:rise_idx_I_u[1]] * V_u[rise_idx_I_u[0]:rise_idx_I_u[1]])
            P_v = np.mean(I_v[rise_idx_I_u[0]:rise_idx_I_u[1]] * V_v[rise_idx_I_u[0]:rise_idx_I_u[1]])
            P_w = np.mean(I_w[rise_idx_I_u[0]:rise_idx_I_u[1]] * V_w[rise_idx_I_u[0]:rise_idx_I_u[1]])
        else: # Otherwise, calculate over the whole block
            P_u = np.mean(I_u * V_u)
            P_v = np.mean(I_v * V_v)
            P_w = np.mean(I_w * V_w)
            
        P_tot = P_u + P_v + P_w
        time_p = time_arr[0] + (time_arr[-1] - time_arr[0]) / 2
        return time_p, P_u, P_v, P_w, P_tot

    def _getCommandVelo(self):
        """Returns a function that slices the ideal velocity curve."""
        time_axis, velocity = self._genCommandFreq()
        
        def slicer(t_start: float, t_end: float):
            i0 = np.searchsorted(time_axis, t_start, side="left")
            i1 = np.searchsorted(time_axis, t_end, side="left")
            return time_axis[i0:i1], velocity[i0:i1]
            
        return slicer

