"""
GhostGPU — rigorous evaluation.

This script exists to answer the three hardest questions a judge can ask:

  Q1  "Why do you need machine learning? Wouldn't a threshold do?"
      -> BASELINE COMPARISON: we run five simpler detectors on identical data
         and report where each one fails.

  Q2  "You claim Stage 1 catches failure types it has never seen. Prove it."
      -> LEAVE-ONE-FAILURE-TYPE-OUT: train with a failure type completely
         removed, then test only on that unseen type.

  Q3  "Is 99% a fluke of one lucky split?"
      -> MULTI-SEED: repeat the whole experiment over several random splits
         and report mean +/- standard deviation.

It also reports the SUSPICIOUS alert rate on healthy jobs, because a system
that constantly cries "suspicious" is operationally noisy even if it never
issues a false GHOST.

Run:  python src/evaluate.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from xgboost import XGBClassifier

import config as C
from features import FEATURE_COLS

warnings.filterwarnings("ignore")

GHOST_PROB = 0.70     # probability at which a window counts as anomalous
STREAK = 2            # consecutive anomalous windows required for a GHOST verdict
STEP = 5              # window stride used when replaying a job


# ------------------------------------------------------------------ utilities
def bootstrap_ci(successes, total, n_boot=4000, seed=0):
    """95% confidence interval for a rate, by bootstrap resampling.

    Reporting a bare '0% false alarms' from a handful of jobs is misleading:
    the interval shows how much uncertainty that number really carries.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    obs = np.zeros(total, dtype=int)
    obs[:successes] = 1
    draws = rng.choice(obs, size=(n_boot, total), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def split_by_job(feats, seed):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(feats.job_id.unique()))
    rng.shuffle(ids)
    n_test = int(len(ids) * 0.25)
    test_ids = set(ids[:n_test].tolist())
    return (feats[~feats.job_id.isin(test_ids)].copy(),
            feats[feats.job_id.isin(test_ids)].copy())


def job_level(test, scores, jobs_meta, threshold=GHOST_PROB, streak=STREAK):
    """
    Replay each test job window-by-window and apply the same hysteresis the live
    system uses. Returns detection rate, false-alarm rate and latency.
    """
    t = test.copy()
    t["score"] = scores
    detected, latencies, false_alarms, n_ghost, n_healthy = 0, [], 0, 0, 0

    for job_id, g in t.groupby("job_id"):
        g = g.sort_values("window_start")
        meta = jobs_meta.loc[job_id]
        is_ghost = meta.label == "ghost"
        onset = meta.ghost_onset
        n_ghost += is_ghost
        n_healthy += (not is_ghost)

        run, fired_at = 0, None
        for _, row in g.iterrows():
            run = run + 1 if row["score"] >= threshold else 0
            if run >= streak:
                fired_at = row["window_start"] + 30   # window end = "now"
                break

        if is_ghost and fired_at is not None:
            detected += 1
            latencies.append(max(0, fired_at - onset))
        elif (not is_ghost) and fired_at is not None:
            false_alarms += 1

    return dict(
        detection_rate=detected / max(n_ghost, 1),
        false_alarm_rate=false_alarms / max(n_healthy, 1),
        median_latency=float(np.median(latencies)) if latencies else float("nan"),
        n_ghost_jobs=n_ghost, n_healthy_jobs=n_healthy,
        n_detected=detected, n_false_alarms=false_alarms,
    )


def window_metrics(y_true, scores, threshold=GHOST_PROB):
    pred = (np.asarray(scores) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return dict(precision=p, recall=r, f1=f1, auc=auc)


# ------------------------------------------------------------------ detectors
def s_util_threshold(train, test):
    """Baseline 1: the classic rule -- 'alert if GPU utilisation drops'."""
    # low utilisation => suspected dead. Scored as a pseudo-probability.
    return (test["gpu_mean"] < 50).astype(float).to_numpy()


def s_flatness_rule(train, test):
    """Baseline 2: 'alert if the utilisation signal goes flat'."""
    return (test["gpu_std"] < 2.0).astype(float).to_numpy()


def s_multi_rule(train, test):
    """Baseline 3: hand-written multi-signal rule an engineer might write."""
    cond = ((test["loss_change"].abs() < 0.01) &
            (test["minutes_since_ckpt"] >= 25) &
            (test["gpu_mean"] > 70))
    cond = cond | (test["nan_ratio"] > 0.5)
    return cond.astype(float).to_numpy()


def s_iso_only(train, test):
    """Baseline 4: Isolation Forest alone (unsupervised), min-max scaled."""
    Xtr = train[FEATURE_COLS].to_numpy(float)
    ytr = train["label"].to_numpy(int)
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=C.SEED, n_jobs=-1).fit(Xtr[ytr == 0])
    a_tr = -iso.score_samples(Xtr)
    a_te = -iso.score_samples(test[FEATURE_COLS].to_numpy(float))
    lo, hi = a_tr.min(), a_tr.max()
    return np.clip((a_te - lo) / max(hi - lo, 1e-9), 0, 1)


def s_xgb_only(train, test):
    """Baseline 5: XGBoost alone, without the anomaly-score feature."""
    Xtr = train[FEATURE_COLS].to_numpy(float)
    ytr = train["label"].to_numpy(int)
    w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    m = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08,
                      subsample=0.9, colsample_bytree=0.9, scale_pos_weight=w,
                      eval_metric="logloss", random_state=C.SEED, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(test[FEATURE_COLS].to_numpy(float))[:, 1]


def s_two_stage(train, test):
    """GhostGPU: Isolation Forest anomaly score fed into XGBoost."""
    Xtr = train[FEATURE_COLS].to_numpy(float)
    ytr = train["label"].to_numpy(int)
    Xte = test[FEATURE_COLS].to_numpy(float)

    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=C.SEED, n_jobs=-1).fit(Xtr[ytr == 0])
    Xtr2 = np.column_stack([Xtr, -iso.score_samples(Xtr)])
    Xte2 = np.column_stack([Xte, -iso.score_samples(Xte)])

    w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    m = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08,
                      subsample=0.9, colsample_bytree=0.9, scale_pos_weight=w,
                      eval_metric="logloss", random_state=C.SEED, n_jobs=-1)
    m.fit(Xtr2, ytr)
    return m.predict_proba(Xte2)[:, 1]


DETECTORS = [
    ("GPU-utilisation threshold", s_util_threshold),
    ("Utilisation-flatness rule", s_flatness_rule),
    ("Hand-written multi-signal rule", s_multi_rule),
    ("Isolation Forest only", s_iso_only),
    ("XGBoost only", s_xgb_only),
    ("GhostGPU (IsoForest + XGBoost)", s_two_stage),
]


# ------------------------------------------------------------ experiment runs
def experiment_baselines(feats, jobs_meta, seeds=(1, 2, 3, 4, 5)):
    print("\n" + "=" * 78)
    print("Q1  BASELINE COMPARISON  —  why machine learning is needed")
    print("=" * 78)
    print("Mean +/- std over", len(seeds), "random job-level splits\n")
    print(f"{'Detector':<32}{'Job recall':>13}{'False alarm':>14}{'Latency':>12}")
    print("-" * 78)

    results = {}
    for name, fn in DETECTORS:
        det, fa, lat = [], [], []
        for sd in seeds:
            tr, te = split_by_job(feats, sd)
            sc = fn(tr, te)
            jl = job_level(te, sc, jobs_meta)
            det.append(jl["detection_rate"])
            fa.append(jl["false_alarm_rate"])
            if not np.isnan(jl["median_latency"]):
                lat.append(jl["median_latency"])
        results[name] = (np.mean(det), np.std(det), np.mean(fa), np.std(fa),
                         np.mean(lat) if lat else float("nan"))
        lat_s = f"{np.mean(lat):.0f} min" if lat else "never"
        print(f"{name:<32}{np.mean(det)*100:>8.1f}±{np.std(det)*100:<4.1f}"
              f"{np.mean(fa)*100:>9.1f}±{np.std(fa)*100:<4.1f}{lat_s:>12}")
    print("-" * 78)
    print("Job recall  = % of silently-dead jobs caught")
    print("False alarm = % of healthy jobs wrongly declared GHOST")
    print("Latency     = median minutes between silent death and detection")
    return results


def experiment_unseen(feats, jobs_meta):
    print("\n" + "=" * 78)
    print("Q2  LEAVE-ONE-FAILURE-TYPE-OUT  —  can it catch what it never saw?")
    print("=" * 78)
    print("Each row: that failure type is REMOVED from training, then tested\n")
    print(f"{'Held-out failure type':<26}{'Two-stage':>14}{'XGBoost only':>16}"
          f"{'IsoForest only':>17}")
    print("-" * 78)

    rows = {}
    for gt in C.GHOST_TYPES:
        # training set: healthy windows + ghost windows of OTHER types only
        tr = feats[(feats.label == 0) | (feats.ghost_type != gt)]
        # test set: healthy windows + ghost windows of THIS type only
        te = feats[(feats.label == 0) | (feats.ghost_type == gt)]
        # keep jobs disjoint: hold out a quarter of jobs for testing
        _, te = split_by_job(te, seed=11)
        tr = tr[~tr.job_id.isin(set(te.job_id.unique()))]

        y = te["label"].to_numpy(int)
        r = {}
        for label, fn in [("two", s_two_stage), ("xgb", s_xgb_only),
                          ("iso", s_iso_only)]:
            sc = fn(tr, te)
            r[label] = window_metrics(y, sc)["recall"]
        rows[gt] = r
        print(f"{gt:<26}{r['two']*100:>12.1f}%{r['xgb']*100:>15.1f}%"
              f"{r['iso']*100:>16.1f}%")
    print("-" * 78)
    print("Recall on the unseen failure type (window level).")
    return rows


def experiment_noise(feats, jobs_meta, seeds=(1, 2, 3, 4, 5)):
    print("\n" + "=" * 78)
    print("Q3  OPERATIONAL NOISE  —  how often does it bother a healthy job?")
    print("=" * 78)
    susp, ghost_fa = [], []
    for sd in seeds:
        tr, te = split_by_job(feats, sd)
        sc = s_two_stage(tr, te)
        t = te.copy()
        t["score"] = sc
        healthy_ids = [j for j in t.job_id.unique()
                       if jobs_meta.loc[j].label == "healthy"]
        h = t[t.job_id.isin(healthy_ids)]
        # SUSPICIOUS = a single window above 0.40 (no streak required)
        susp.append((h.groupby("job_id")["score"].max() >= 0.40).mean())
        jl = job_level(te, sc, jobs_meta)
        ghost_fa.append(jl["false_alarm_rate"])
    print(f"  healthy jobs that ever showed SUSPICIOUS : "
          f"{np.mean(susp)*100:.1f}% ± {np.std(susp)*100:.1f}")
    print(f"  healthy jobs wrongly declared GHOST      : "
          f"{np.mean(ghost_fa)*100:.1f}% ± {np.std(ghost_fa)*100:.1f}")
    print("\n  A GHOST verdict requires a sustained streak, which is why the")
    print("  GHOST false-alarm rate is far below the SUSPICIOUS rate.")
    return np.mean(susp), np.mean(ghost_fa)


def experiment_headline(feats, jobs_meta, seeds=(1, 2, 3, 4, 5)):
    """Pooled result across seeds, with bootstrap confidence intervals."""
    print("\n" + "=" * 78)
    print("HEADLINE  —  GhostGPU, pooled over", len(seeds), "splits, with 95% CIs")
    print("=" * 78)
    det = fa = n_g = n_h = 0
    lats = []
    for sd in seeds:
        tr, te = split_by_job(feats, sd)
        sc = s_two_stage(tr, te)
        jl = job_level(te, sc, jobs_meta)
        det += jl["n_detected"]; n_g += jl["n_ghost_jobs"]
        fa += jl["n_false_alarms"]; n_h += jl["n_healthy_jobs"]
        if not np.isnan(jl["median_latency"]):
            lats.append(jl["median_latency"])

    r_lo, r_hi = bootstrap_ci(det, n_g)
    f_lo, f_hi = bootstrap_ci(fa, n_h)
    print(f"  ghost jobs detected      : {det}/{n_g} = {det/n_g*100:.1f}%"
          f"   95% CI [{r_lo*100:.1f}%, {r_hi*100:.1f}%]")
    print(f"  healthy jobs false GHOST : {fa}/{n_h} = {fa/n_h*100:.1f}%"
          f"   95% CI [{f_lo*100:.1f}%, {f_hi*100:.1f}%]")
    print(f"  median detection latency : {np.mean(lats):.0f} min")
    print("\n  Evidence scope: controlled synthetic failure injection.")
    print(f"  Failure timing: {'FAVOURABLE (>=%d min of runtime after failure)' % C.MIN_GHOST_RUNTIME if C.MIN_GHOST_RUNTIME else 'HARD (a job may die shortly before it would have ended)'}")
    return det, n_g, fa, n_h


def main():
    feats = pd.read_csv(C.DATA_DIR / "features.csv")
    jobs_meta = pd.read_csv(C.JOBS_CSV).set_index("job_id")

    print("GhostGPU — rigorous evaluation")
    print(f"  jobs        : {len(jobs_meta)}")
    print(f"  windows     : {len(feats):,}")
    print(f"  ghost types : {C.GHOST_TYPES}")
    print("  NOTE: all results are on SYNTHETIC failure injections, not on")
    print("        real production silent-failure labels.")

    experiment_headline(feats, jobs_meta)
    experiment_baselines(feats, jobs_meta)
    experiment_unseen(feats, jobs_meta)
    experiment_noise(feats, jobs_meta)

    print("\n" + "=" * 78)
    print("Evaluation complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
