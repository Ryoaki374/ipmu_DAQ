import dataclasses

from pathlib import Path
import numpy as np

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# ------------------------------ Constants ------------------------------
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent # scripting directory
else:
    BASE_DIR = Path.cwd()

CONFIG_PATH: Path = BASE_DIR / "_config_preset.toml"
CONFIG_RUN_PATH: Path = BASE_DIR / "_config_preset.toml"
RUNS_DIR: Path = BASE_DIR / "../runs"

# ------------------------------ Configs ------------------------------

@dataclasses.dataclass
class IOConfig:
    sample_rate: int
    gen_chunk_sec: float
    proc_interval: float
    queue_depth: int
    quad_depth: int

@dataclasses.dataclass
class GUIConfig:
    display_sec: float
    plot_sec: float
    gui_interval_ms: int
    pruning: int

@dataclasses.dataclass
class LoggingConfig:
    log_chunk: int
    log_data_num: int

@dataclasses.dataclass
class EncoderPostProcConfig:
    quadpulse_width: float
    threshold: float

@dataclasses.dataclass
class DebugEncoderConfig:
    input_velocity: float
    pulse_height: float
    pulse_duty: float
    pulse_phase_A: float

@dataclasses.dataclass
class DebugPowerConfig:
    amplitude: float
    phase: float

@dataclasses.dataclass
class DriverConfig:
    device_com: str
    target_speed_rps: float
    current_init: int
    electrical_angle_init: int
    t_DC: int
    pps: int
    step: int
    rst: float
    ppsps: float
    t_stablerot: int
    t_excess_spindown: int
    dir_rotation: int
    t_current_reduction_duration: int
    step_current_reduction: int

@dataclasses.dataclass
class DependentConfig:
    # io
    chunk_sec: float
    n_samples_gen: int
    samples_proc: int
    # gui
    history: float
    count_history: int
    velo_history: int
    pow_history: int
    # logging
    # encoder_postproc
    # debug_mock
    num_pulses: int
    stable_sec: int
    rel_axis_mock: float
    # debug encoder
    pulse_width: float
    pulse_phase_B: float
    # debug power
    omega: float

@dataclasses.dataclass
class AppConfig:
    io: IOConfig
    gui: GUIConfig
    logging: LoggingConfig
    encoder_postproc: EncoderPostProcConfig
    debug_encoder: DebugEncoderConfig
    debug_power: DebugPowerConfig
    driver: DriverConfig
    # --
    dependent: DependentConfig

    @staticmethod
    def fromDict(preset: dict, run: dict) -> "AppConfig":
        io = preset["io"]; gui = preset["gui"]; lg = preset["logging"]
        ep = preset["encoder_postproc"]; de = preset["debug_encoder"]; dp = preset["debug_power"]
        drv = run["driver"] if "driver" in run else preset["driver"]

        # dependent
        chunk_sec = io["gen_chunk_sec"]
        n_samples_gen = int(io["sample_rate"] * io["gen_chunk_sec"]) # 100_000 * 0.05 = 5000
        samples_proc = int(io["sample_rate"] * io["proc_interval"]) # 0.125s * 100kHz = 12500
        history = int(io["sample_rate"] * gui["plot_sec"]/gui["pruning"]) # = 15000
        count_history = int(1 / io["proc_interval"] * 10) # <-- 1/0.125 * 10 sec
        velo_history = count_history
        pow_history = count_history
        #--
        num_pulses = 36000
        stable_sec = 7200
        rel_axis_mock = (np.arange(n_samples_gen, dtype=np.float32) / io["sample_rate"])
        #--
        pulse_width = 1 / (de["input_velocity"] * 512)  # [s]
        pulse_phase_B = -pulse_width / 4
        #--
        omega = 2 * np.pi * 36 * de["input_velocity"]

        dependent = DependentConfig(
            chunk_sec=chunk_sec,
            n_samples_gen=n_samples_gen,
            samples_proc=samples_proc,
            history=history,
            count_history=count_history,
            velo_history=velo_history,
            pow_history=pow_history,
            num_pulses=num_pulses,
            stable_sec=stable_sec,
            rel_axis_mock=rel_axis_mock,
            pulse_width=pulse_width,
            pulse_phase_B=pulse_phase_B,
            omega=omega,
        )

        return AppConfig(
            io=IOConfig(**io),
            gui=GUIConfig(**gui),
            logging=LoggingConfig(**lg),
            encoder_postproc=EncoderPostProcConfig(**ep),
            debug_encoder=DebugEncoderConfig(**de),
            debug_power=DebugPowerConfig(**dp),
            driver=DriverConfig(**drv),
            dependent=dependent,
        )

# ------------------------------ App ------------------------------
def _loadConfig(path: Path = CONFIG_PATH) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)

def printConfig(cfg: AppConfig):
    for name, data_obj in cfg.__dict__.items():
        if not dataclasses.is_dataclass(data_obj):
            continue
        header = name.replace('_', ' ').title()
        print(f"\n[{header}]")
        for field in dataclasses.fields(data_obj):
            key = field.name
            value = getattr(data_obj, key)
            if isinstance(value, np.ndarray):
                print(f"  {key:<20}: ndarray(shape={value.shape}, dtype={value.dtype})")
            else:
                if isinstance(value, float):
                    print(f"  {key:<20}: {value:.6f}")
                else:
                    print(f"  {key:<20}: {value}")

def initParams(config_preset: dict, config_run: dict, debug: bool = True, runs_dir: Path | str = RUNS_DIR) -> None:
    cfg = AppConfig.fromDict(config_preset, config_run)
    if debug:
        printConfig(cfg)
    return cfg