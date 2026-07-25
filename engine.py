"""
GhostGPU — central configuration.

Every tunable number lives here so the rest of the code stays clean, and so
you can adapt the project quickly on-site (new ghost type, new threshold, etc.).
"""

from pathlib import Path

# ----- folders -----
ROOT = Path(__file__).resolve().parent.parent      # the ghostgpu/ folder
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TELEMETRY_CSV = DATA_DIR / "telemetry.csv"          # one row per job per minute
JOBS_CSV = DATA_DIR / "jobs.csv"                     # one row per job (answer key)

# ----- data-generation settings -----
SEED = 42
N_JOBS = 240                 # how many training jobs to simulate
GHOST_FRACTION = 0.30        # ~30% of jobs silently die. Kept deliberately low so
                             # there are enough HEALTHY jobs to estimate the
                             # false-alarm rate with a usable confidence interval.
MIN_MINUTES = 60             # shortest job length (in "minutes")

# How much runtime a ghost job is guaranteed AFTER it silently dies.
#   0  = HARD mode  -> a job may die shortly before it would have finished, so
#                      there is not always time to detect it. This is realistic
#                      and produces the honest, lower recall number.
#  60  = FAVOURABLE  -> every failure has >= 60 min of ghost time. Detection is
#                      much easier; quoting only this number overstates recall.
# We report BOTH so the operating envelope is explicit.
MIN_GHOST_RUNTIME = 0
MAX_MINUTES = 240            # longest job length
CHECKPOINT_EVERY = 15        # a healthy job writes a checkpoint every N minutes

MODEL_FAMILIES = ["unet3d", "resnet", "vit", "unet2d", "efficientnet"]

# The ways a job can silently die. Adding a new ghost type later is as simple
# as adding its name here and a matching branch in generate_data.apply_ghost().
GHOST_TYPES = ["nan_loss", "frozen", "stalled"]

# ----- cost model (used later for the "money burned" counter) -----
GPU_COST_PER_HOUR = 2.0      # $ per GPU-hour (change to your own number)
