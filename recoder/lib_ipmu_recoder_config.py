"""Shared acquisition and quadrature-processing constants for ``recoder``."""

# Change only this value to switch between the planned DAQ sample rates.
SAMPLING_RATE = 10_000
if SAMPLING_RATE not in (10_000, 1_000_000):
    raise ValueError("SAMPLING_RATE must be 10_000 or 1_000_000 Hz")

PROCESS_INTERVAL = 0.2
GEN_CHUNK_SEC = 0.2

# Values carried over from src/_config_preset.toml.
ENCODER_THRESHOLD = 2.5
QUAD_PULSE_WIDTH = 0.00025
PULSE_HEIGHT = 5.0
MOCK_INPUT_VELOCITY = 1.0

# Four edge counts are produced for each encoder pulse period.
ENCODER_PULSES_PER_REVOLUTION = 512
ENCODER_COUNTS_PER_REVOLUTION = ENCODER_PULSES_PER_REVOLUTION * 4

HDF5_COLUMN_NAMES = (
    "time",
    "pulse_A",
    "pulse_B",
    "pulse_C",
    "pulse_D",
)
HDF5_COLUMN_COUNT = len(HDF5_COLUMN_NAMES)
HDF5_CHUNK_SIZE = 1024

SAMPLES_PER_PROCESS = int(SAMPLING_RATE * PROCESS_INTERVAL)
SAMPLES_PER_GENERATED_CHUNK = int(SAMPLING_RATE * GEN_CHUNK_SEC)
