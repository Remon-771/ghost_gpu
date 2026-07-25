"""
GhostGPU — Step 3: the AI core (two-stage detection model).

STAGE 1 — Isolation Forest (unsupervised)
    Trained ONLY on windows from healthy jobs. Learns what "normal training"
    looks like and scores how unfamiliar each new window is. Because it never
    sees failure labels, it is our intended route to failure types the model has
    not been trained on. (See evaluate.py — that ability is measured, not assumed.)

STAGE 2 — XGBoost (supervised)
    Trained on labelled windows plus the Stage-1 anomaly score, then wrapped in
    PLATT CALIBRATION so the output is a genuine probability rather than an
    arbitrary score. Calibration quality is reported (Brier score + expected
    calibration error), so "calibrated" is a claim we can defend.

THREE-WAY SPLIT, BY JOB
    train (60%) -> fits the models
    calib (20%) -> fits the Platt calibrator AND selects the operating point
    test  (20%) -> touched once, for reporting only
No job ever appears in more than one split, so no window can leak between them.

Run:      python src/train_model.py
Outputs:  models/iso_forest.pkl, models/xgb.pkl, models/calibrator.pkl,
          models/meta.json
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, classification_report,
                             confusion_matrix, roc_auc_score)
from xgboost import XGBClassifier

import config as C
from features import FEATURE_COLS

MODELS_DIR = C.ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def three_way_split(feats, seed=C.SEED):
    """Split whole jobs into train / calibration / test."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(feats.job_id.unique()))
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(n * 0.20)
    n_cal = int(n * 0.20)
    test_ids = set(ids[:n_test].tolist())
    cal_ids = set(ids[n_test:n_test + n_cal].tolist())
    train = feats[~feats.job_id.isin(test_ids | cal_ids)].copy()
    calib = feats[feats.job_id.isin(cal_ids)].copy()
    test = feats[feats.job_id.isin(test_ids)].copy()
    assert not (set(train.job_id) & set(calib.job_id) & set(test.job_id))
    return train, calib, test


def expected_calibration_error(y, p, bins=10):
    """How far predicted probabilities drift from observed frequencies."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def job_level_sweep(calib, probs, jobs_meta):
    """
    Pick the operating point (probability threshold + streak length) on the
    CALIBRATION split -- never on test. We prefer the setting that maximises
    job recall while keeping the healthy false-GHOST rate at zero.
    """
    t = calib.copy()
    t["p"] = probs
    best = None
    for thr in np.arange(0.50, 0.96, 0.05):
        for streak in (2, 3, 4):
            det = fa = n_g = n_h = 0
            for job_id, g in t.groupby("job_id"):
                g = g.sort_values("window_start")
                is_ghost = jobs_meta.loc[job_id].label == "ghost"
                n_g += is_ghost
                n_h += (not is_ghost)
                run = 0
                fired = False
                for p in g["p"]:
                    run = run + 1 if p >= thr else 0
                    if run >= streak:
                        fired = True
                        break
                det += (is_ghost and fired)
                fa += ((not is_ghost) and fired)
            recall = det / max(n_g, 1)
            far = fa / max(n_h, 1)
            score = recall - 3.0 * far          # false alarms are costly
            if best is None or score > best[0]:
                best = (score, float(thr), int(streak), recall, far)
    return dict(ghost_prob=best[1], ghost_streak=best[2],
                calib_recall=best[3], calib_false_alarm=best[4])


def main():
    feats = pd.read_csv(C.DATA_DIR / "features.csv")
    jobs_meta = pd.read_csv(C.JOBS_CSV).set_index("job_id")
    train, calib, test = three_way_split(feats)

    Xtr = train[FEATURE_COLS].to_numpy(float); ytr = train["label"].to_numpy(int)
    Xca = calib[FEATURE_COLS].to_numpy(float); yca = calib["label"].to_numpy(int)
    Xte = test[FEATURE_COLS].to_numpy(float);  yte = test["label"].to_numpy(int)

    # ---------- Stage 1: Isolation Forest, healthy windows only ----------
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=C.SEED, n_jobs=-1).fit(Xtr[ytr == 0])
    a = lambda X: -iso.score_samples(X)
    Xtr2 = np.column_stack([Xtr, a(Xtr)])
    Xca2 = np.column_stack([Xca, a(Xca)])
    Xte2 = np.column_stack([Xte, a(Xte)])

    # ---------- Stage 2: XGBoost ----------
    w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=w,
                        eval_metric="logloss", random_state=C.SEED, n_jobs=-1)
    xgb.fit(Xtr2, ytr)

    # ---------- Platt calibration, fitted on the calibration split ----------
    raw_ca = xgb.predict_proba(Xca2)[:, 1]
    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(raw_ca.reshape(-1, 1), yca)
    cal = lambda raw: platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    p_ca = cal(raw_ca)
    p_te = cal(xgb.predict_proba(Xte2)[:, 1])

    # ---------- operating point chosen on calibration, not on test ----------
    op = job_level_sweep(calib, p_ca, jobs_meta)

    # ---------- report ----------
    pred = (p_te >= op["ghost_prob"]).astype(int)
    auc = roc_auc_score(yte, p_te)
    brier = brier_score_loss(yte, p_te)
    ece = expected_calibration_error(yte, p_te)
    cm = confusion_matrix(yte, pred)

    print("GhostGPU — model trained")
    print(f"  split (jobs)         : train {train.job_id.nunique()} | "
          f"calib {calib.job_id.nunique()} | test {test.job_id.nunique()}  (disjoint)")
    print(f"  windows              : {len(train):,} / {len(calib):,} / {len(test):,}")
    print(f"\n  operating point chosen on CALIBRATION split:")
    print(f"    ghost probability  >= {op['ghost_prob']:.2f}")
    print(f"    consecutive windows = {op['ghost_streak']}")
    print(f"    calib job recall   = {op['calib_recall']*100:.1f}%   "
          f"false alarm = {op['calib_false_alarm']*100:.1f}%")

    print(f"\n  TEST split (touched once):")
    print(f"    ROC-AUC            : {auc:.3f}")
    print(f"    Brier score        : {brier:.4f}   (lower is better)")
    print(f"    Calibration error  : {ece:.4f}   (ECE, lower is better)")
    print(f"\n  confusion matrix (rows true, cols predicted)")
    print(f"      healthy  ghost\n   H   {cm[0,0]:5d}  {cm[0,1]:5d}\n   G   {cm[1,0]:5d}  {cm[1,1]:5d}")
    print("\n" + classification_report(yte, pred, target_names=["healthy", "ghost"],
                                       digits=3, zero_division=0))

    test2 = test.assign(pred=pred)
    gh = test2[test2.label == 1]
    if len(gh):
        print("  recall per ghost type (window level):")
        for gt, grp in gh.groupby("ghost_type"):
            print(f"    {gt:<10} {grp.pred.mean():.3f}  ({len(grp)} windows)")

    imps = sorted(zip(FEATURE_COLS + ["anomaly_score"], xgb.feature_importances_),
                  key=lambda kv: kv[1], reverse=True)[:6]
    print("\n  top signals:")
    for n_, i_ in imps:
        print(f"    {n_:<20} {i_:.3f}")

    # ---------- save ----------
    for name, obj in [("iso_forest", iso), ("xgb", xgb), ("calibrator", platt)]:
        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    meta = dict(
        feature_cols=FEATURE_COLS,
        evidence_scope="controlled synthetic failure injection; not production data",
        split=dict(strategy="three-way, by job, disjoint",
                   train_jobs=int(train.job_id.nunique()),
                   calib_jobs=int(calib.job_id.nunique()),
                   test_jobs=int(test.job_id.nunique())),
        operating_point=op,
        test_metrics=dict(roc_auc=float(auc), brier=float(brier),
                          calibration_error=float(ece)),
        calibration="Platt scaling fitted on a held-out calibration split",
        ghost_runtime_mode=("favourable (>=%d min after failure)" % C.MIN_GHOST_RUNTIME
                            if C.MIN_GHOST_RUNTIME else "hard (failure may occur near job end)"),
        top_features=[{"name": n_, "importance": float(i_)} for n_, i_ in imps],
    )
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  saved                : iso_forest.pkl, xgb.pkl, calibrator.pkl, meta.json")


if __name__ == "__main__":
    main()
