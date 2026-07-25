"""
GhostGPU — Step 2: feature extraction.

Raw telemetry (one row per minute) is too noisy to judge directly.
Instead we slide a WINDOW over each job and, for every window, compute a small
set of meaningful signals — the "vital signs" of that stretch of training:

  LEARNING PROGRESS   loss_slope, loss_change, loss_std, nan_ratio
                      -> is the model actually learning?
  UTILIZATION RHYTHM  gpu_mean, gpu_std, gpu_range, gpu_flatness
                      -> healthy training wobbles; ghosts go flat
  OUTPUT ACTIVITY     ckpt_count, minutes_since_ckpt
                      -> is progress still being saved?
  CONTEXT             elapsed_min  (minutes since the job started)

NOTE ON LEAKAGE: an earlier version used `job_age_frac = window_end / total_job_
length`. Total length is only known AFTER a job finishes, so that feature leaked
future information and could not be computed live. It is replaced by
`elapsed_min`, which is available at every moment of a running job.

A window is labelled `1` (ghost) if it lies entirely AFTER the job's ghost onset,
`0` if the job is healthy or the window is fully before onset. Windows that
straddle the onset are dropped, so the model learns from clean examples.

Run:      python src/features.py
Outputs:  data/features.csv
"""

import numpy as np
import pandas as pd

import config as C

WINDOW = 30   # minutes per window
STEP = 5      # slide the window forward this many minutes


def window_features(win: pd.DataFrame) -> dict:
    """Compute one feature row from one window of telemetry."""
    loss = win["loss"].to_numpy(dtype=float)
    gpu = win["gpu_util"].to_numpy(dtype=float)
    mem = win["mem_util"].to_numpy(dtype=float)
    ckpt = win["checkpoint"].to_numpy(dtype=int)

    # --- learning progress ---
    nan_ratio = float(np.isnan(loss).mean())
    valid = loss[~np.isnan(loss)]
    if len(valid) >= 3:
        x = np.arange(len(valid))
        loss_slope = float(np.polyfit(x, valid, 1)[0])   # negative = improving
        loss_change = float(valid[-1] - valid[0])
        loss_std = float(valid.std())
        loss_mean = float(valid.mean())
    else:
        # nearly all NaN: a dead-obvious ghost signature
        loss_slope, loss_change, loss_std, loss_mean = 0.0, 0.0, 0.0, 0.0

    # --- utilization rhythm ---
    gpu_mean = float(gpu.mean())
    gpu_std = float(gpu.std())
    gpu_range = float(gpu.max() - gpu.min())
    # flatness: how little the signal moves minute-to-minute (ghosts are flat)
    gpu_flatness = float(1.0 / (1.0 + np.abs(np.diff(gpu)).mean())) if len(gpu) > 1 else 1.0
    mem_std = float(mem.std())

    # --- output activity ---
    ckpt_count = int(ckpt.sum())
    if ckpt_count > 0:
        minutes_since_ckpt = int(len(ckpt) - 1 - np.max(np.where(ckpt == 1)[0]))
    else:
        minutes_since_ckpt = int(len(ckpt))

    return dict(
        loss_slope=loss_slope, loss_change=loss_change, loss_std=loss_std,
        loss_mean=loss_mean, nan_ratio=nan_ratio,
        gpu_mean=gpu_mean, gpu_std=gpu_std, gpu_range=gpu_range,
        gpu_flatness=gpu_flatness, mem_std=mem_std,
        ckpt_count=ckpt_count, minutes_since_ckpt=minutes_since_ckpt,
    )


FEATURE_COLS = [
    "loss_slope", "loss_change", "loss_std", "loss_mean", "nan_ratio",
    "gpu_mean", "gpu_std", "gpu_range", "gpu_flatness", "mem_std",
    "ckpt_count", "minutes_since_ckpt", "elapsed_min",
]


def build(telemetry: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    onset_map = dict(zip(jobs.job_id, jobs.ghost_onset))
    label_map = dict(zip(jobs.job_id, jobs.label))
    type_map = dict(zip(jobs.job_id, jobs.ghost_type))

    for job_id, g in telemetry.groupby("job_id", sort=True):
        g = g.sort_values("minute").reset_index(drop=True)
        n = len(g)
        onset = onset_map[job_id]
        is_ghost_job = label_map[job_id] == "ghost"

        for start in range(0, n - WINDOW + 1, STEP):
            end = start + WINDOW          # window covers [start, end)
            win = g.iloc[start:end]

            if is_ghost_job:
                if end <= onset:
                    y = 0                  # entirely before the death
                elif start >= onset:
                    y = 1                  # entirely after the death
                else:
                    continue               # straddles onset -> skip (ambiguous)
            else:
                y = 0

            feats = window_features(win)
            feats.update(
                job_id=int(job_id), window_start=int(start),
                elapsed_min=float(end),   # causal: known while running
                label=int(y), ghost_type=type_map[job_id],
            )
            rows.append(feats)

    return pd.DataFrame(rows)


def main():
    telemetry = pd.read_csv(C.TELEMETRY_CSV)
    jobs = pd.read_csv(C.JOBS_CSV)
    feats = build(telemetry, jobs)
    out = C.DATA_DIR / "features.csv"
    feats.to_csv(out, index=False)

    print("GhostGPU — features built")
    print(f"  window / step       : {WINDOW} min / {STEP} min")
    print(f"  windows total       : {len(feats):,}")
    print(f"  healthy / ghost win : {int((feats.label == 0).sum()):,} / "
          f"{int((feats.label == 1).sum()):,}")
    print(f"  features per window : {len(FEATURE_COLS)}")
    print(f"  saved               : {out.name}")


if __name__ == "__main__":
    main()
