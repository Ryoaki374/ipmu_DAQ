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

CONFIG_PATH: Path = BASE_DIR / "_config_run.toml"
RUNS_DIR: Path = BASE_DIR / "../runs"

# ------------------------------ Configs ------------------------------
@dataclasses.dataclass
class DriverConfig:
    device_com: str
    target_speed_rps: float
    current_init: 20
    electrical_angle_init: 200
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
    target_speed_rpm: int
    electrical_frequency: float
    pps_fin: int
    t_total: int

@dataclasses.dataclass
class AppConfig:
    driver: DriverConfig
    dependent: DependentConfig

    @staticmethod
    def fromDict(run: dict) -> "AppConfig":
        drv = run["driver"]

        t_DC = drv["t_DC"]
        step = drv["step"] 
        rst = drv["rst"]
        pps = drv["pps"]
        t_stablerot = drv["t_stablerot"]
        t_excess_spindown = drv["t_excess_spindown"]

        # dependent
        target_speed_rpm = int(60*drv["target_speed_rps"])
        electrical_frequency = 36*drv["target_speed_rps"]
        pps_fin = int(electrical_frequency*1000*0.994)
        # Total rotation time (=RUN_SEC) # added on 2025/05/27 by Taisei
        t_total = t_DC + (pps_fin-pps) / (step / rst) + t_stablerot + t_excess_spindown

        dependent = DependentConfig(
            target_speed_rpm = target_speed_rpm,
            electrical_frequency = electrical_frequency,
            pps_fin = pps_fin,
            t_total = t_total,
        )

        return AppConfig(
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

def initParams(config_run: dict, debug: bool = True, runs_dir: Path | str = RUNS_DIR) -> None:
    cfg = AppConfig.fromDict(config_run)
    if debug:
        printConfig(cfg)
    return cfg