"""
GhostGPU — Step 4a: the detection engine.

Wraps the two trained models into something the rest of the system can use:

    engine.score(job_id, window_df)  ->  verdict dict

It does three things beyond raw prediction:

  1. HYSTERESIS — a single odd window is not enough. A job must look anomalous
     for several consecutive windows before it is called a GHOST. This is what
     stops the dashboard flip-flopping and keeps false alarms low.

  2. EXPLANATION — every verdict says *why* in plain English, e.g.
     "loss has stopped improving; no checkpoint for 30 min".

  3. BURN ACCOUNTING — once a job is judged a ghost, we keep counting the
     GPU-minutes, money and energy it has wasted since the ghost began.

Thresholds live in one place (ENGINE_SETTINGS) so they can be changed live —
useful when an unexpected problem is thrown at you on the day.
"""

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C
from features import FEATURE_COLS, window_features

MODELS_DIR = C.ROOT / "models"

# ---- everything tunable, in one place (editable at runtime via the API) ----
ENGINE_SETTINGS = {
    "suspicious_prob": 0.40,   # >= this -> SUSPICIOUS
    "ghost_prob": 0.70,        # >= this -> counts as an anomalous window
    "ghost_streak": 2,         # this many anomalous windows in a row -> GHOST
    "gpu_cost_per_hour": C.GPU_COST_PER_HOUR,
    "gpu_watts": 300,          # for the energy estimate
}


@dataclass
class JobState:
    """What the engine remembers about one job between windows."""
    job_id: int
    streak: int = 0
    verdict: str = "HEALTHY"
    ghost_since_minute: int | None = None
    last_prob: float = 0.0
    history: list = field(default_factory=list)   # recent (minute, prob) pairs


class Engine:
    def __init__(self):
        with open(MODELS_DIR / "iso_forest.pkl", "rb") as f:
            self.iso = pickle.load(f)
        with open(MODELS_DIR / "xgb.pkl", "rb") as f:
            self.xgb = pickle.load(f)
        # Platt calibrator, fitted on a held-out calibration split, so the number
        # we show is a probability rather than an arbitrary score.
        cal_path = MODELS_DIR / "calibrator.pkl"
        self.calibrator = None
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                self.calibrator = pickle.load(f)
        # operating point selected on the calibration split (never on test)
        meta_path = MODELS_DIR / "meta.json"
        if meta_path.exists():
            import json
            op = json.loads(meta_path.read_text()).get("operating_point", {})
            if "ghost_prob" in op:
                ENGINE_SETTINGS["ghost_prob"] = float(op["ghost_prob"])
            if "ghost_streak" in op:
                ENGINE_SETTINGS["ghost_streak"] = int(op["ghost_streak"])
        self.states: dict[int, JobState] = {}

    # ------------------------------------------------------------------ utils
    def settings(self):
        return dict(ENGINE_SETTINGS)

    def update_settings(self, **kwargs):
        """Change thresholds at runtime (used by the /settings endpoint)."""
        for k, v in kwargs.items():
            if k in ENGINE_SETTINGS and v is not None:
                ENGINE_SETTINGS[k] = type(ENGINE_SETTINGS[k])(v)
        return self.settings()

    def reset(self, job_id: int | None = None):
        if job_id is None:
            self.states.clear()
        else:
            self.states.pop(job_id, None)

    # ------------------------------------------------------------- prediction
    def probability(self, window_df: pd.DataFrame,
                    elapsed_min: float = 0.0) -> tuple[float, dict]:
        """Ghost probability for one 30-minute window, plus its raw features."""
        feats = window_features(window_df)
        feats["elapsed_min"] = float(elapsed_min)
        x = np.array([[feats.get(c, 0.0) for c in FEATURE_COLS]], dtype=float)
        anomaly = float(-self.iso.score_samples(x)[0])
        x2 = np.column_stack([x, [[anomaly]]])
        raw = float(self.xgb.predict_proba(x2)[0, 1])
        if self.calibrator is not None:
            prob = float(self.calibrator.predict_proba(np.array([[raw]]))[0, 1])
        else:
            prob = raw
        feats["anomaly_score"] = anomaly
        return prob, feats

    def explain(self, feats: dict) -> str:
        """Turn the numbers into a sentence a human can act on."""
        reasons = []
        if feats.get("nan_ratio", 0) > 0.3:
            reasons.append("loss has become NaN")
        elif abs(feats.get("loss_change", 0)) < 0.005:
            reasons.append("loss has stopped improving")
        if feats.get("gpu_std", 99) < 1.5 and feats.get("gpu_mean", 0) > 80:
            reasons.append("GPU is busy but its rhythm is flat")
        if feats.get("minutes_since_ckpt", 0) >= 25:
            reasons.append(f"no checkpoint for {int(feats['minutes_since_ckpt'])} min")
        if not reasons:
            reasons.append("training signals look normal")
        return "; ".join(reasons)

    # ---------------------------------------------------------------- verdict
    def score(self, job_id: int, window_df: pd.DataFrame,
              current_minute: int, gpus: int = 1,
              elapsed_min: float = 0.0) -> dict:
        st = self.states.setdefault(job_id, JobState(job_id=job_id))
        prob, feats = self.probability(window_df, elapsed_min=elapsed_min)
        s = ENGINE_SETTINGS

        # streak logic (hysteresis)
        if prob >= s["ghost_prob"]:
            st.streak += 1
        else:
            st.streak = 0

        if st.streak >= s["ghost_streak"]:
            verdict = "GHOST"
            if st.ghost_since_minute is None:
                # the ghost really began when the streak started
                window_len = len(window_df)
                st.ghost_since_minute = max(
                    0, current_minute - window_len * (s["ghost_streak"] - 1))
        elif prob >= s["suspicious_prob"]:
            verdict = "SUSPICIOUS"
        else:
            verdict = "HEALTHY"
            st.ghost_since_minute = None

        st.verdict = verdict
        st.last_prob = prob
        st.history.append((current_minute, round(prob, 3)))
        st.history = st.history[-60:]

        # burn accounting
        wasted_min = 0
        if verdict == "GHOST" and st.ghost_since_minute is not None:
            wasted_min = max(0, current_minute - st.ghost_since_minute)
        gpu_hours = wasted_min / 60.0 * gpus
        cost = gpu_hours * s["gpu_cost_per_hour"]
        energy_kwh = gpu_hours * s["gpu_watts"] / 1000.0

        return dict(
            job_id=job_id, minute=current_minute,
            verdict=verdict, ghost_prob=round(prob, 3),
            streak=st.streak,
            reason=self.explain(feats),
            ghost_since_minute=st.ghost_since_minute,
            wasted_minutes=wasted_min,
            wasted_gpu_hours=round(gpu_hours, 2),
            wasted_cost=round(cost, 2),
            wasted_kwh=round(energy_kwh, 2),
            history=list(st.history),
        )


_engine: Engine | None = None


def get_engine() -> Engine:
    """Load the engine once and reuse it."""
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine
