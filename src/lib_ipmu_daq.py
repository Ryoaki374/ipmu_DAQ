from dataclasses import dataclass

import numpy as np
import time
import queue
import threading
import h5py, pathlib
from datetime import datetime
from collections import deque

import pyqtgraph as pg
from PyQt6 import QtCore


def initState():
    global DEBUG
    DEBUG = False


def initParamsPostProcess(_config_preset, DEBUG=True):
    # [io]
    global SAMPLE_RATE, GEN_CHUNK_SEC, PROC_INTERVAL, QUEUE_DEPTH, QUAD_DEPTH
    SAMPLE_RATE = _config_preset["io"]["sample_rate"]  # samplerate for DAQ device [Hz]
    GEN_CHUNK_SEC = _config_preset["io"]["gen_chunk_sec"]
    PROC_INTERVAL = _config_preset["io"]["proc_interval"]
    QUEUE_DEPTH = _config_preset["io"]["queue_depth"]
    QUAD_DEPTH = _config_preset["io"]["quad_depth"]
    # also update dependent parameters
    global CHUNK_SEC, N_SAMPLES_GEN, SAMPLES_PROC
    CHUNK_SEC = GEN_CHUNK_SEC
    N_SAMPLES_GEN = int(SAMPLE_RATE * GEN_CHUNK_SEC)  # 100_000 * 0.05 = 5000
    SAMPLES_PROC = int(PROC_INTERVAL * SAMPLE_RATE) # 0.125s * 100kHz = 12500

    # [gui]
    global DISPLAY_SEC, PLOT_SEC, GUI_INTERVAL_MS, PRUNING, HISTORY, COUNT_HISTORY, VELO_HISTORY, POW_HISTORY
    DISPLAY_SEC = _config_preset["gui"]["display_sec"]
    PLOT_SEC = _config_preset["gui"]["plot_sec"]
    GUI_INTERVAL_MS = _config_preset["gui"]["gui_interval_ms"]
    PRUNING = _config_preset["gui"]["pruning"]
    # also update dependent parameters
    HISTORY = int(SAMPLE_RATE * PLOT_SEC / PRUNING)  # = 15000
    COUNT_HISTORY = int(1 / PROC_INTERVAL * 10)  # <-- 1/0.125 * 10 sec
    VELO_HISTORY = COUNT_HISTORY
    POW_HISTORY = COUNT_HISTORY

    # [logging]
    global LOG_CHUNK, LOG_DATA_NUM
    LOG_CHUNK = _config_preset["logging"]["log_chunk"]
    LOG_DATA_NUM = _config_preset["logging"]["log_data_num"]

    # [encoder_postproc]
    global QUADPULSE_WIDTH, THRESHOLD_DEFAULT, IDEAL_CPS
    QUADPULSE_WIDTH = _config_preset["encoder_postproc"]["quadpulse_width"]
    THRESHOLD_DEFAULT = _config_preset["encoder_postproc"]["threshold"]
    IDEAL_CPS = _config_preset["debug_encoder"]["input_velocity"]

    # [debug_mock]
    global NUM_PULSES, STABLE_SEC, REL_AXIS_MOCK
    NUM_PULSES = 36000
    STABLE_SEC = 7200
    REL_AXIS_MOCK = (np.arange(N_SAMPLES_GEN, dtype=np.float32) / SAMPLE_RATE)  # 0 ... 0.2 s

    # [debug_encoder]
    global PULSE_HEIGHT, INPUT_VELOCITY, PULSE_WIDTH, PULSE_DUTY, PULSE_PHASE_A, PULSE_PHASE_B
    INPUT_VELOCITY = _config_preset["debug_encoder"]["input_velocity"]
    PULSE_HEIGHT = _config_preset["debug_encoder"]["pulse_height"]
    PULSE_WIDTH = 1 / (INPUT_VELOCITY * 512)  # [s]
    PULSE_DUTY = _config_preset["debug_encoder"]["pulse_duty"]
    PULSE_PHASE_A = _config_preset["debug_encoder"]["phase_A"]
    PULSE_PHASE_B = -PULSE_WIDTH / 4  # phase offset [s]

    # [debug_power]
    global AMPLITUDE, OMEGA
    AMPLITUDE = _config_preset["debug_power"]["amplitude"]
    OMEGA = 2 * np.pi * 36 * INPUT_VELOCITY

    if DEBUG:
        print(f"--- [lib_ipmu_daq] parameters initialized ---")
        print(f"SAMPLE_RATE = {SAMPLE_RATE} [Hz]")
        print(f"GEN_CHUNK_SEC = {GEN_CHUNK_SEC} [s]")
        print(f"PROC_INTERVAL = {PROC_INTERVAL} [s]")
        print(f"QUEUE_DEPTH = {QUEUE_DEPTH}")
        print(f"QUAD_DEPTH = {QUAD_DEPTH}")
        print(f"CHUNK_SEC = {CHUNK_SEC} [s]")
        print(f"N_SAMPLES_GEN = {N_SAMPLES_GEN} [samples]")
        print(f"SAMPLES_PROC = {SAMPLES_PROC} [samples]")
        print(f"DISPLAY_SEC = {DISPLAY_SEC} [s]")
        print(f"PLOT_SEC = {PLOT_SEC} [s]")
        print(f"GUI_INTERVAL_MS = {GUI_INTERVAL_MS} [ms]")
        print(f"PRUNING = {PRUNING}")
        print(f"HISTORY = {HISTORY} [points]")
        print(f"COUNT_HISTORY = {COUNT_HISTORY} [points]")
        print(f"VELO_HISTORY = {VELO_HISTORY} [points]")
        print(f"POW_HISTORY = {POW_HISTORY} [points]")
        print(f"LOG_CHUNK = {LOG_CHUNK} [chunks]")
        print(f"LOG_DATA_NUM = {LOG_DATA_NUM} [data]")
        print(f"QUADPULSE_WIDTH = {QUADPULSE_WIDTH} [s]")
        print(f"THRESHOLD_DEFAULT = {THRESHOLD_DEFAULT} [V]")
        print(f"IDEAL_CPS = {IDEAL_CPS} [cps]")
        print(f"NUM_PULSES = {NUM_PULSES} [pulses]")
        print(f"STABLE_SEC = {STABLE_SEC} [s]")
        print(f"PULSE_HEIGHT = {PULSE_HEIGHT} [V]")
        print(f"INPUT_VELOCITY = {INPUT_VELOCITY} [rps]")
        print(f"PULSE_WIDTH = {PULSE_WIDTH} [s]")
        print(f"PULSE_DUTY = {PULSE_DUTY} [frac]")
        print(f"PULSE_PHASE_A = {PULSE_PHASE_A} [s]")
        print(f"PULSE_PHASE_B = {PULSE_PHASE_B} [s]")
        print(f"AMPLITUDE = {AMPLITUDE} [V]")
        print(f"OMEGA = {OMEGA} [rad/s]")


def initBuffers():
    global buf_q, quad_q
    buf_q = queue.Queue(maxsize=QUEUE_DEPTH)  # raw (t, A, B)
    quad_q = queue.Queue(maxsize=QUAD_DEPTH)  # processed (t, A, B, quad)

def initThreads():
    global stop_writer
    stop_writer = threading.Event()

def initStorer():
    global dset, h5f
    run_dir = pathlib.Path("../runs")
    run_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    h5f = h5py.File(f"{run_dir}/{timestamp}.h5", "w")
    dset = h5f.create_dataset(
        "log", shape=(0, LOG_DATA_NUM), maxshape=(None, LOG_DATA_NUM),
        dtype=np.float32, chunks=(LOG_CHUNK, LOG_DATA_NUM), compression="gzip"
    )
    print("dset shape  =", dset.shape)
    print("dset maxshape =", dset.maxshape)

def initParamsDriver(_config_run):
    global t_DC, pps, step, rst, target_freq
    t_DC = _config_run["driver"]["t_DC"]
    pps = _config_run["driver"]["pps"]
    step = _config_run["driver"]["step"]
    rst = _config_run["driver"]["rst"]
    target_freq = _config_run["driver"]["target_freq"]


def genChunkPulse(
    t: np.ndarray,
    *,
    height: float = PULSE_HEIGHT,
    width: float = PULSE_WIDTH,
    duty: float = PULSE_DUTY,
    phase: float = 0.0,
) -> np.ndarray:
    mod = (t + phase) % width
    return np.where(mod < duty * width, height, 0.0).astype(np.float32)

def genChunkSin(
    time: np.ndarray,
    *,
    A: float = AMPLITUDE,  # amplitude
    omega: float = OMEGA,  # angular frequency
    phase: float = 0.0,  # phase offset in radians
) -> np.ndarray:
    n = np.random.randn(len(time))  # random noise
    return A * np.sin(omega * time + phase) + A * 0.01 * n

def getPulseDirection(
    dA: np.ndarray,
    dB: np.ndarray,
    *,
    threshold: float,
    prev_A: bool | None = None,
    prev_B: bool | None = None,
) -> tuple[np.ndarray, bool, bool]:
    # --- current logic level ----------------------------------------
    A = dA > threshold
    B = dB > threshold

    # --- previous sample for XOR -----------------------------------
    if prev_A is None:  # first block → old behaviour
        A_prev = np.concatenate(([A[0]], A[:-1]))
        B_prev = np.concatenate(([B[0]], B[:-1]))
    else:  # use states carried over from last block
        A_prev = np.concatenate(([prev_A], A[:-1]))
        B_prev = np.concatenate(([prev_B], B[:-1]))

    dir_log = (B_prev ^ A).astype(int) - (A_prev ^ B).astype(int)

    return dir_log.astype(np.int8), bool(A[-1]), bool(B[-1])


def getPulseCount(dir_log: np.ndarray) -> int:
    return np.sum(dir_log)


def genQuadPulse(
    t: np.ndarray, dir_log: np.ndarray, width: float, height: float, sampling_rate: int
) -> np.ndarray:
    samples = int(width * sampling_rate)
    if samples <= 0:
        return np.zeros_like(dir_log, dtype=np.float32)
    base = np.full(samples, height, dtype=np.float32)
    return np.convolve(dir_log, base, mode="full")[: len(t)]

# ------------------------------------------------------------  producer thread
def genMockSignals() -> None:
    chunk_idx = 0
    next_t = time.perf_counter()
    while not stop_writer.is_set():
        base = chunk_idx * GEN_CHUNK_SEC
        t_axis = REL_AXIS_MOCK + base
        pulse_A = genChunkPulse(t_axis, phase=PULSE_PHASE_A)
        pulse_B = genChunkPulse(t_axis, phase=PULSE_PHASE_B)

        # --- add sine wave for testing purposes ---
        Iu = genChunkSin(t_axis, A=0.3, omega=OMEGA, phase=0.0)
        Iv = genChunkSin(t_axis, A=0.3, omega=OMEGA, phase=2 * np.pi / 3)
        Iw = genChunkSin(t_axis, A=0.3, omega=OMEGA, phase=4 * np.pi / 3)
        Vu = genChunkSin(t_axis, A=1.0, omega=OMEGA, phase=np.pi / 8)
        Vv = genChunkSin(t_axis, A=1.0, omega=OMEGA, phase=2 * np.pi / 3 + np.pi / 8)
        Vw = genChunkSin(t_axis, A=1.0, omega=OMEGA, phase=4 * np.pi / 3 + np.pi / 8)

        try:
            buf_q.put_nowait((t_axis, pulse_A, pulse_B, Iu, Iv, Iw, Vu, Vv, Vw))
        except queue.Full:
            pass

        chunk_idx += 1
        next_t += GEN_CHUNK_SEC
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.perf_counter()


def genCommandFreq(t_DC, pps, step, rst, num_pulse, target_freq, t_stable, proc_interval):
    num_step = np.ceil(
        (num_pulse - pps) * target_freq / step
    )  # stableに到達するまでの段数
    time = np.arange(
        0, t_DC + num_step * rst + t_stable, proc_interval, dtype=np.float32
    )
    freq = np.zeros(
        int((t_DC + num_step * rst + t_stable) * (1 / proc_interval)), dtype=np.float32
    )
    n_DC = int(np.round(t_DC / proc_interval))  # DCの速度指令値=0rps
    freq[:n_DC] = 0.0
    n_rst = int(np.round(rst / proc_interval))  # 加速中の速度指令値
    freq[n_DC : int(n_DC + n_rst * num_step)] = [
        pps / num_pulse + target_freq / num_pulse * step * (i + 1)
        for i in range(int(num_step))
        for j in freq[int(n_DC + i * n_rst) : int(n_DC + (i + 1) * n_rst)]
    ]
    # Stableの速度指令値
    freq[int(n_DC + n_rst * num_step) :] = target_freq
    return time, freq

def utilSchmittTrigger(upper, lower, current): # schmitt trigger
    y_schmitt = np.zeros_like(current)
    state = 0.0
    for i, sample in enumerate(current):
        if state == 0.0 and sample >= upper:
            state = 1.0
        elif state == 1.0 and sample <= lower:
            state = 0.0
        y_schmitt[i] = state

    d = np.diff(y_schmitt.astype(int))  # diff（+1 なら立ち上がり, −1 なら立ち下がり）
    rise_idx = np.where(d == 1)[0] + 1  # 立ち上がり位置
    fall_idx = np.where(d == -1)[0] + 1  # 立ち下がり位置

    return rise_idx, fall_idx

def getPower(time, I_u, I_v, I_w, V_u, V_v, V_w, upper, lower):
    # aquire rise index of I_u, I_v, I_w within single period of the signal
    rise_idx_I_u = utilSchmittTrigger(upper, lower, I_u)[0]
    # rise_idx_I_v = utilSchmittTrigger(upper, lower, I_v)[0]
    # rise_idx_I_w = utilSchmittTrigger(upper, lower, I_w)[0]

    # proc_intervalの中にI_uが2周期以上入るとき->初めの1周期を取り出してPを計算
    if len(rise_idx_I_u) >= 3:
        period = rise_idx_I_u[1] - rise_idx_I_u[0]
        P_u = np.mean(
            I_u[rise_idx_I_u[0] : rise_idx_I_u[0] + period]
            * V_u[rise_idx_I_u[0] : rise_idx_I_u[0] + period]
        )
        P_v = np.mean(
            I_v[
                rise_idx_I_u[0]
                + int(period / 3) : rise_idx_I_u[0]
                + int(4 * period / 3)
            ]
            * V_v[
                rise_idx_I_u[0]
                + int(period / 3) : rise_idx_I_u[0]
                + int(4 * period / 3)
            ]
        )
        P_w = np.mean(
            I_w[
                rise_idx_I_u[0]
                + int(2 * period / 3) : rise_idx_I_u[0]
                + int(5 * period / 3)
            ]
            * V_w[
                rise_idx_I_u[0]
                + int(2 * period / 3) : rise_idx_I_u[0]
                + int(5 * period / 3)
            ]
        )
        P_tot = P_u + P_v + P_w

    # proc_intervalの中にI_uが2周期入らないとき->proc_interval全体でPを計算
    else:
        P_u = np.mean(I_u * V_u)
        P_v = np.mean(I_v * V_v)
        P_w = np.mean(I_w * V_w)
        P_tot = P_u + P_v + P_w

    # 中央の時間を計算
    time_p = time[0] + (time[-1] - time[0]) / 2

    return time_p, P_u, P_v, P_w, P_tot


def getCommandVelo(t_DC, pps, step, rst, num_pulse, target_freq, t_stable, proc_interval):
    time_axis, velocity = genCommandFreq(
        t_DC, pps, step, rst, num_pulse, target_freq, t_stable, proc_interval
    )

    def slicer(t_start: float, t_end: float):
        i0 = np.searchsorted(time_axis, t_start, side="left")
        i1 = np.searchsorted(time_axis, t_end,   side="left")
        return time_axis[i0:i1], velocity[i0:i1]

    return slicer


### proceccsor and gui
def processor() -> None:
    # ---------- one-time init ----------
    global ideal_provider
    if 'ideal_provider' not in globals() or ideal_provider is None:
        ideal_provider = getCommandVelo(t_DC, pps, step, rst, NUM_PULSES, target_freq, STABLE_SEC, PROC_INTERVAL)

    # ---------- logging ----------
    buf = np.empty((LOG_CHUNK, LOG_DATA_NUM), dtype=np.float32)  # 内部バッファ
    buf_len = 0
    print("buf shape =", buf.shape)

    # --- Python deque リングバッファ ---
    ring_t: deque[np.float32] = deque(); ring_a: deque[np.float32] = deque(); ring_b: deque[np.float32] = deque()
    ring_Iu: deque[np.float32] = deque(); ring_Iv: deque[np.float32] = deque(); ring_Iw: deque[np.float32] = deque()
    ring_Vu: deque[np.float32] = deque(); ring_Vv: deque[np.float32] = deque(); ring_Vw: deque[np.float32] = deque()


    last_A = last_B = None

    next_proc = time.perf_counter()
    cum_count = 0
    last_ts   = next_proc

    while not stop_writer.is_set():
        # ---------- 非ブロッキングで出来るだけ取り込む ----------
        try:
            while True:
                t, pA, pB, Iu, Iv, Iw, Vu, Vv, Vw = buf_q.get_nowait()
                ring_t.extend(t); ring_a.extend(pA); ring_b.extend(pB)
                ring_Iu.extend(Iu); ring_Iv.extend(Iv); ring_Iw.extend(Iw)
                ring_Vu.extend(Vu); ring_Vv.extend(Vv); ring_Vw.extend(Vw)
                buf_q.task_done()
        except queue.Empty:
            pass

        # ---------- 次の処理時刻まで待機 ----------
        now = time.perf_counter()
        if now < next_proc:
            time.sleep(next_proc - now)
            continue
        next_proc += PROC_INTERVAL

        # ---------- サンプル不足ならスキップ ----------
        if len(ring_t) < SAMPLES_PROC:
            continue

        # ---------- deque → NumPy へコピー ----------
        t_blk = np.array([ring_t.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)
        a_blk = np.array([ring_a.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)
        b_blk = np.array([ring_b.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)

        Iu_blk = np.array([ring_Iu.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)
        Iv_blk = np.array([ring_Iv.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)
        Iw_blk = np.array([ring_Iw.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)
        Vu_blk = np.array([ring_Vu.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)
        Vv_blk = np.array([ring_Vv.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)
        Vw_blk = np.array([ring_Vw.popleft() for _ in range(SAMPLES_PROC)], dtype=np.float32)

        # ---------- 信号処理 ----------
        dir_log, last_A, last_B = getPulseDirection(
            a_blk, b_blk,
            threshold=THRESHOLD_DEFAULT,
            prev_A=last_A, prev_B=last_B
        )
        quad_sig  = genQuadPulse(t_blk, dir_log, QUADPULSE_WIDTH, PULSE_HEIGHT, SAMPLE_RATE)
        delta_cnt = getPulseCount(dir_log)
        cum_count += delta_cnt
        velocity  = delta_cnt / PROC_INTERVAL / 2048



        # Power calculation
        time_p, P_u, P_v, P_w, P_tot = getPower(
            t_blk, Iu_blk, Iv_blk, Iw_blk,
            Vu_blk, Vv_blk, Vw_blk, 0, -0.1
        )

        # ---------- 理想速度スライス ----------
        #t_ref, v_ref = ideal_provider.slice(t_blk[0], t_blk[-1])
        t_ref, v_ref = ideal_provider(t_blk[0], t_blk[-1])

        # ---------- ログ ----------
        '''if DEBUG:
            now = time.perf_counter()
            jitter = (now - last_ts) * 1e3
            logger.info(
                "EPOCH = %f, wall = %6.2f ms, jitter = %6.2f ms  delta c=%+d, v=%6.3f, len(dir_log)=%d, buf_len=%d",
                now, jitter, (now - last_ts) * 1e3,
                delta_cnt, velocity, len(dir_log), buf_len
            )'''

        #if DEBUG:
        #    now = time.perf_counter()
        #    jitter = (now - last_ts) * 1e3
        #    logger.info(
        #        "EPOCH = %f, wall = %6.2f ms, jitter = %6.2f ms  delta c=%+d, v=%6.3f, v_ref=%6.3f, time_p = %f, P_tot = %f",
        #        now, jitter, (now - last_ts) * 1e3,
        #        delta_cnt, velocity, v_ref, time_p, P_tot
        #    )
        #    last_ts = now

        # ---------- GUI へ送信 ----------
        try:
            quad_q.put_nowait(
                (t_blk, a_blk, b_blk, quad_sig,
                 t_blk[-1], cum_count, velocity,
                 t_ref, v_ref, time_p, P_tot, Iu_blk, Vu_blk)
            )
        except queue.Full:
            pass

         # ---- append to buffer ----
        buf[buf_len] = (t_blk[-1], (v_ref[-1] if v_ref.size else 0.0), velocity, P_tot)
        buf_len += 1
        if buf_len == buf.shape[0]:
            n = dset.shape[0]
            dset.resize(n+buf_len, axis=0); dset[-buf_len:] = buf
            buf_len = 0

     # ---- final flush ----
    if buf_len:
        n = dset.shape[0]
        dset.resize(n+buf_len, axis=0); dset[-buf_len:] = buf[:buf_len]
    h5f.close()

def start_gui() -> None:
    pg.setConfigOptions(useOpenGL=True, background="w", foreground="k")
    app = pg.mkQApp("Live plots")

    win = pg.GraphicsLayoutWidget(title="DEMO")

    layout = win.ci.layout                 # GraphicsLayout の中身
    layout.setColumnStretchFactor(0, 4)
    layout.setColumnStretchFactor(1, 5)
    #win_sig = pg.GraphicsLayoutWidget(show=True, title="Signal")
    #win_vel = pg.GraphicsLayoutWidget(show=True, title="Velocity")

    #win_sig.resize(800, 600)
    #win_vel.resize(800, 600)
    #win_sig.show()

    pruning = PRUNING
    if pruning < 1:
        pruning = 1 # 0以下の値は無効とし、1に設定


    win.resize(800, 600)
    win.show()

    # [0, 0] A/B -------------------------------------------------------
    plt_ab = win.addPlot(row=0, col=0, title="RAW A / B")
    curve_A = plt_ab.plot(pen=pg.mkPen("#ff4b00", width=3),stepMode="right")
    curve_B = plt_ab.plot(pen=pg.mkPen("#005aff", width=3),stepMode="right")
    plt_ab.setLabel("left", "Amplitude [V]")
    plt_ab.setLabel("bottom", "Time [s]")
    plt_ab.setYRange(-0.5, PULSE_HEIGHT + 0.5)

    # [2,0] I/V waveform --------------------------------------------
    plt_IV = win.addPlot(row=2, col=0, title="RAW I / V")
    curve_I = plt_IV.plot(pen=pg.mkPen("r", width=3))
    curve_V = plt_IV.plot(pen=pg.mkPen("b", width=3))
    plt_IV.setLabel("left", "Amplitude [a.u.]")
    plt_IV.setLabel("bottom", "Time [s]")
    plt_IV.setYRange(-1.2, 1.2)

    # [0,1] count (fixed x-axis) -------------------------------------
    plt_cnt = win.addPlot(row=0, col=1, title="Velovity - Command")
    curve_cnt = plt_cnt.plot(pen=pg.mkPen("#03af7a", width=3))
    #plt_cnt.setXRange(0, RUN_SEC, padding=0)
    #plt_cnt.enableAutoRange("x", True)
    plt_cnt.setLabel("left", "Diff")
    plt_cnt.setLabel("bottom", "Time [s]")

    # [1,1] velocity (fixed x-axis) ----------------------------------
    plt_vel = win.addPlot(row=1, col=1, title="Velocity")
    curve_vel     = plt_vel.plot(pen=pg.mkPen("#00a0e9", width=3))   # measured
    curve_vel_ref = plt_vel.plot(pen=pg.mkPen("m", width=3),stepMode="right")   # ideal (new)
    #plt_vel.setXRange(0, RUN_SEC, padding=0)
    #plt_vel.enableAutoRange("x", True)
    plt_vel.setLabel("left", "Velocity [rps]")
    plt_vel.setLabel("bottom", "Time [s]")

    # [2,1] power ----------------------------------
    plt_pow = win.addPlot(row=2, col=1, title="Power")
    curve_pow     = plt_pow.plot(pen=pg.mkPen("#f6aa00", width=3))   # measured
    plt_pow.setLabel("left", "Power [W]")
    plt_pow.setLabel("bottom", "Time [s]")

    # buffers ---------------------------------------------------------
    xs = ya = yb = yq = np.empty(0, dtype=np.float32)
    xs_cnt = y_cnt = np.empty(0, dtype=np.float32)
    xs_vel = y_vel = np.empty(0, dtype=np.float32)
    xr = yr = np.empty(0, dtype=np.float32)
    y_Iu = y_Vu = np.empty(0, dtype=np.float32)
    xs_time_p = y_P_tot = np.empty(0, dtype=np.float32)


def start_gui() -> None:
    pg.setConfigOptions(useOpenGL=True, background="w", foreground="k")
    app = pg.mkQApp("Live plots")

    win = pg.GraphicsLayoutWidget(title="DEMO")

    layout = win.ci.layout                 # GraphicsLayout の中身
    layout.setColumnStretchFactor(0, 4)
    layout.setColumnStretchFactor(1, 5)
    #win_sig = pg.GraphicsLayoutWidget(show=True, title="Signal")
    #win_vel = pg.GraphicsLayoutWidget(show=True, title="Velocity")

    #win_sig.resize(800, 600)
    #win_vel.resize(800, 600)
    #win_sig.show()
    pruning = PRUNING
    if pruning < 1:
        pruning = 1 # 0以下の値は無効とし、1に設定


    win.resize(800, 600)
    win.show()

    # [0, 0] A/B -------------------------------------------------------
    plt_ab = win.addPlot(row=0, col=0, title="RAW A / B")
    curve_A = plt_ab.plot(pen=pg.mkPen("#ff4b00", width=3),stepMode="right")
    curve_B = plt_ab.plot(pen=pg.mkPen("#005aff", width=3),stepMode="right")
    plt_ab.setLabel("left", "Amplitude [V]")
    plt_ab.setLabel("bottom", "Time [s]")
    plt_ab.setYRange(-0.5, PULSE_HEIGHT + 0.5)

    # [1,0] Quad waveform --------------------------------------------
    #plt_q = win.addPlot(row=1, col=0, title="Quad pulse")
    #curve_Q = plt_q.plot(pen=pg.mkPen("m", width=3))
    #plt_q.setLabel("left", "Amplitude [V]")
    #plt_q.setLabel("bottom", "Time [s]")
    #plt_q.setYRange(-PULSE_HEIGHT - 0.5, PULSE_HEIGHT + 0.5)

    # [2,0] I/V waveform --------------------------------------------
    plt_IV = win.addPlot(row=2, col=0, title="PhaseU I / V")
    curve_I = plt_IV.plot(pen=pg.mkPen("r", width=3))
    curve_V = plt_IV.plot(pen=pg.mkPen("b", width=3))
    plt_IV.setLabel("left", "Amplitude [a.u.]")
    plt_IV.setLabel("bottom", "Time [s]")
    plt_IV.setYRange(-1.2, 1.2)

    # [0,1] count (fixed x-axis) -------------------------------------
    plt_cnt = win.addPlot(row=0, col=1, title=r"Delta velocity")
    curve_cnt = plt_cnt.plot(pen=pg.mkPen("#03af7a", width=3))
    #plt_cnt.setXRange(0, RUN_SEC, padding=0)
    #plt_cnt.enableAutoRange("x", True)
    plt_cnt.setLabel("left", "Diff")
    plt_cnt.setLabel("bottom", "Time [s]")

    # [1,1] velocity
    plt_vel = win.addPlot(row=1, col=1, title="Velocity")
    curve_vel     = plt_vel.plot(pen=pg.mkPen("#00a0e9", width=3))   # measured
    curve_vel_ref = plt_vel.plot(pen=pg.mkPen("#a05aff", width=3), stepMode="right")   # ideal (new)
    #plt_vel.setXRange(0, RUN_SEC, padding=0)
    #plt_vel.enableAutoRange("x", True)
    plt_vel.setLabel("left", "Velocity [rps]")
    plt_vel.setLabel("bottom", "Time [s]")

    # [2,1] power ----------------------------------
    plt_pow = win.addPlot(row=2, col=1, title="Power")
    curve_pow     = plt_pow.plot(pen=pg.mkPen("#f6aa00", width=3))   # measured
    plt_pow.setLabel("left", "Power [W]")
    plt_pow.setLabel("bottom", "Time [s]")

    # buffers ---------------------------------------------------------
    xs = ya = yb = yq = np.empty(0, dtype=np.float32)
    xs_cnt = y_cnt = np.empty(0, dtype=np.float32)
    xs_vel = y_vel = np.empty(0, dtype=np.float32)
    xr = yr = np.empty(0, dtype=np.float32)       # ideal velocity buffers (new)
    y_Iu = y_Vu = np.empty(0, dtype=np.float32)  # I/V buffers (new)
    xs_time_p = y_P_tot = np.empty(0, dtype=np.float32)  # time and power buffers (new)

    def refresh():
        nonlocal xs, ya, yb, yq, xs_cnt, y_cnt, xs_vel, y_vel, xr, yr, xs_time_p, y_P_tot, y_Iu, y_Vu
        try:
            while True:
                # receive processed data
                t_ax, pA, pB, qsig, t_end, cum_cnt, vel, t_ref, v_ref, time_p, P_tot, Iu_blk, Vu_blk = quad_q.get_nowait()

                xs = np.concatenate((xs, t_ax[::pruning]))[-HISTORY:]
                ya = np.concatenate((ya, pA[::pruning]))[-HISTORY:]
                yb = np.concatenate((yb, pB[::pruning]))[-HISTORY:]
                #yq = np.concatenate((yq, qsig))[-HISTORY:]

                xs_cnt = np.append(xs_cnt, t_end)[-COUNT_HISTORY:]
                y_cnt  = np.append(y_cnt, vel-v_ref)[-COUNT_HISTORY:]

                xs_vel = np.append(xs_vel, t_end)[-VELO_HISTORY:]
                y_vel  = np.append(y_vel, vel)[-VELO_HISTORY:]

                xr = np.concatenate((xr, t_ref))[-VELO_HISTORY:]
                yr = np.concatenate((yr, v_ref))[-VELO_HISTORY:]

                y_Iu = np.concatenate((y_Iu, Iu_blk[::pruning]))[-HISTORY:]
                y_Vu = np.concatenate((y_Vu, Vu_blk[::pruning]))[-HISTORY:]

                xs_time_p = np.append(xs_time_p, time_p)[-POW_HISTORY:]
                y_P_tot = np.append(y_P_tot, P_tot)[-POW_HISTORY:]

                quad_q.task_done()
        except queue.Empty:
            pass

        # scrolling window for waveforms only
        if xs.size:
            start = xs[-1] - PLOT_SEC
            plt_ab.setXRange(start, xs[-1], padding=0)
            plt_IV.setXRange(start, xs[-1], padding=0)

        # --- auto-range for count/velocity ---
        if xs_cnt.size:
            plt_cnt.setXRange(xs_cnt[-1]-10, xs_cnt[-1], padding=0)

        if xs_vel.size:
            plt_vel.setXRange(xs_vel[-1]-10, xs_vel[-1], padding=0)

        if xs_time_p.size:
            plt_pow.setXRange(xs_time_p[-1]-10, xs_time_p[-1], padding=0)

        # --- push data to curves ---
        curve_A.setData(xs, ya)
        curve_B.setData(xs, yb)
        curve_I.setData(xs, y_Iu)
        curve_V.setData(xs, y_Vu)
        curve_cnt.setData(xs_cnt, y_cnt)
        curve_vel.setData(xs_vel, y_vel)
        curve_vel_ref.setData(xr, yr)
        curve_pow.setData(xs_time_p, y_P_tot)

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(GUI_INTERVAL_MS)

    # auto-stop after RUN_SEC
    QtCore.QTimer.singleShot(int(RUN_SEC * 1000), lambda: (stop_writer.set(), app.quit()))
    app.exec()

if __name__ == "__main__":
    threading.Thread(target=log_listener, daemon=True).start()  # start log listener

    gen_th  = threading.Thread(target=generator, daemon=True)
    proc_th = threading.Thread(target=processor, daemon=True)

    gen_th.start()
    proc_th.start()
    # con_th.start()

    start_gui()  # blocks until the user closes the window or timer expires

    # join threads and exit
    stop_writer.set()
    gen_th.join()
    proc_th.join()

    print("Graceful shutdown.")