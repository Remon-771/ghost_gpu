"""
GhostGPU — Step 1: realistic synthetic data generator.

We don't have a real GPU cluster, so we simulate one — but deliberately make it
MESSY, the way real clusters are. If ghosts were trivially separable (e.g. "no
checkpoint = ghost"), the model would learn nothing useful and the accuracy
would be a meaningless 100%.

Realism we build in:
  * every job has its own checkpoint interval; some healthy jobs never checkpoint
  * healthy jobs hit LEARNING PLATEAUS where loss barely improves
  * healthy jobs have long VALIDATION PHASES where the GPU goes quiet/flat
  * NaN-ghosts often KEEP writing checkpoints (the loop still runs — it just
    saves garbage weights), so "no checkpoint" is not a reliable tell
  * ghosts occasionally keep a faint wobble, so flatness alone isn't enough

Per-minute signals emitted per job:
    gpu_util, mem_util, loss, checkpoint

Run:      python src/generate_data.py
Outputs:  data/telemetry.csv (per-minute)  +  data/jobs.csv (answer key)
"""

import numpy as np
import pandas as pd

import config as C


def make_healthy(minutes, rng):
    """Per-minute arrays for a healthy — but realistically messy — run."""
    t = np.arange(minutes)

    # loss: exponential decay toward a floor, with noise
    floor = rng.uniform(0.15, 0.5)
    loss = floor + rng.uniform(1.5, 3.0) * np.exp(-t / (minutes * rng.uniform(0.3, 0.6)))
    loss += rng.normal(0, 0.025, minutes)

    # LEARNING PLATEAUS: stretches where loss barely moves. Real plateaus can be
    # as flat as a dead job -- this is what makes the problem genuinely hard.
    for _ in range(rng.integers(1, 4)):
        p_len = int(rng.integers(20, 55))
        p_start = int(rng.integers(0, max(1, minutes - p_len)))
        seg = min(p_len, minutes - p_start)
        loss[p_start:p_start + seg] = (loss[p_start]
                                       + rng.normal(0, rng.uniform(0.0008, 0.004), seg))

    # CONVERGENCE: many healthy runs flatten out completely near the end --
    # loss stops moving even though training is perfectly fine.
    if rng.random() < 0.45:
        c_start = int(minutes * rng.uniform(0.6, 0.8))
        seg = minutes - c_start
        loss[c_start:] = loss[c_start] + rng.normal(0, rng.uniform(0.0008, 0.002), seg)

    # gpu: lively rhythm, per-job amplitude
    amp = rng.uniform(3.5, 8.0)
    base = rng.uniform(78, 92)
    gpu = base + amp * np.sin(t / rng.uniform(2.0, 4.5)) + rng.normal(0, 2.5, minutes)

    # VALIDATION PHASES: GPU dips and goes calm for a while (benign quiet)
    for _ in range(rng.integers(1, 4)):
        v_len = int(rng.integers(4, 14))
        v_start = int(rng.integers(0, max(1, minutes - v_len)))
        gpu[v_start:v_start + v_len] = (base - rng.uniform(25, 45)
                                        + rng.normal(0, 1.2, min(v_len, minutes - v_start)))

    # LONG EVALUATION PHASE (the genuinely hard case): ~30% of healthy jobs run
    # a long eval pass where the training loss does not update and the GPU is
    # steadily busy with uniform work -- i.e. flat loss AND flat GPU, exactly
    # what a dead job looks like. The model must learn to tell these apart.
    if rng.random() < 0.30:
        e_len = int(rng.integers(22, 45))
        e_start = int(rng.integers(0, max(1, minutes - e_len)))
        seg = min(e_len, minutes - e_start)
        loss[e_start:e_start + seg] = loss[e_start] + rng.normal(0, 0.0012, seg)
        gpu[e_start:e_start + seg] = rng.uniform(88, 95) + rng.normal(0, 0.9, seg)
        ckpt_hold = (e_start, e_start + seg)
    else:
        ckpt_hold = None
    gpu = np.clip(gpu, 0, 100)

    mem = np.clip(rng.uniform(55, 82) + rng.normal(0, 3, minutes), 0, 100)

    # checkpoints: per-job interval; ~12% of healthy jobs never checkpoint at all
    ckpt = np.zeros(minutes, dtype=int)
    if rng.random() > 0.12:
        interval = int(rng.integers(10, 31))
        offset = int(rng.integers(0, interval))
        ckpt[offset::interval] = 1
        # occasionally a checkpoint is skipped (disk busy, etc.)
        for i in np.where(ckpt == 1)[0]:
            if rng.random() < 0.15:
                ckpt[i] = 0
    if ckpt_hold is not None:
        ckpt[ckpt_hold[0]:ckpt_hold[1]] = 0

    return gpu, mem, loss, ckpt


def apply_ghost(gpu, mem, loss, ckpt, ghost_type, onset, rng):
    """Corrupt a run from `onset` onward — while keeping the GPU looking busy."""
    minutes = len(loss)
    idx = np.arange(onset, minutes)
    if len(idx) == 0:
        return gpu, mem, loss, ckpt

    if ghost_type == "nan_loss":
        loss[idx] = np.nan
        # IMPORTANT realism: the training loop is still running, so it often
        # keeps saving (garbage) checkpoints. Only sometimes do they stop.
        if rng.random() < 0.45:
            ckpt[idx] = 0

    elif ghost_type == "frozen":
        # data loader hung: loss stuck, rhythm mostly flat (but not always dead flat)
        loss[idx] = loss[onset - 1] + rng.normal(0, rng.uniform(0.0008, 0.003), len(idx))
        wobble = rng.uniform(0.4, 3.2)          # sometimes a strong wobble remains
        gpu[idx] = rng.uniform(86, 94) + rng.normal(0, wobble, len(idx))
        if rng.random() < 0.7:
            ckpt[idx] = 0

    elif ghost_type == "stalled":
        # deadlocked: pinned high, nothing progresses
        loss[idx] = loss[onset - 1] + rng.normal(0, rng.uniform(0.0006, 0.0025), len(idx))
        gpu[idx] = rng.uniform(93, 98) + rng.normal(0, 0.6, len(idx))
        mem[idx] = mem[onset - 1]
        if rng.random() < 0.8:
            ckpt[idx] = 0

    # To add a NEW ghost type: add its name to config.GHOST_TYPES and an
    # `elif ghost_type == "...":` branch here.
    gpu[:] = np.clip(gpu, 0, 100)
    return gpu, mem, loss, ckpt


def main():
    rng = np.random.default_rng(C.SEED)
    tele_rows, job_rows = [], []

    for job_id in range(1, C.N_JOBS + 1):
        minutes = int(rng.integers(C.MIN_MINUTES, C.MAX_MINUTES + 1))
        model = str(rng.choice(C.MODEL_FAMILIES))
        gpu_id = int(rng.integers(0, 8))

        gpu, mem, loss, ckpt = make_healthy(minutes, rng)

        if rng.random() < C.GHOST_FRACTION:
            ghost_type = str(rng.choice(C.GHOST_TYPES))
            onset = int(rng.integers(int(minutes * 0.3), int(minutes * 0.85)))
            if C.MIN_GHOST_RUNTIME > 0:
                # favourable mode: guarantee enough runtime after the failure
                onset = min(onset, max(35, minutes - C.MIN_GHOST_RUNTIME))
            gpu, mem, loss, ckpt = apply_ghost(gpu, mem, loss, ckpt,
                                               ghost_type, onset, rng)
            label, gtype = "ghost", ghost_type
        else:
            label, gtype, onset = "healthy", "none", -1

        job_rows.append(dict(job_id=job_id, model_family=model, gpu_id=gpu_id,
                             minutes=minutes, label=label, ghost_type=gtype,
                             ghost_onset=onset))

        for m in range(minutes):
            tele_rows.append(dict(
                job_id=job_id, minute=m,
                gpu_util=round(float(gpu[m]), 2),
                mem_util=round(float(mem[m]), 2),
                loss=(None if np.isnan(loss[m]) else round(float(loss[m]), 4)),
                checkpoint=int(ckpt[m]),
            ))

    tele = pd.DataFrame(tele_rows)
    jobs = pd.DataFrame(job_rows)
    tele.to_csv(C.TELEMETRY_CSV, index=False)
    jobs.to_csv(C.JOBS_CSV, index=False)

    n_ghost = int((jobs.label == "ghost").sum())
    print("GhostGPU — data generated")
    print(f"  jobs                : {len(jobs)}")
    print(f"  healthy / ghost     : {len(jobs) - n_ghost} / {n_ghost}")
    print("  ghost types         :",
          jobs[jobs.label == 'ghost'].ghost_type.value_counts().to_dict())
    print(f"  telemetry rows      : {len(tele):,}")
    print(f"  saved               : {C.TELEMETRY_CSV.name}, {C.JOBS_CSV.name}")


if __name__ == "__main__":
    main()
