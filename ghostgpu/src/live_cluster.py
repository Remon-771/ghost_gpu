"""
GhostGPU — Step 5: the LIVE cluster simulator.

Instead of replaying a CSV, this keeps a set of jobs "running" in real time.
Every tick advances each job by one simulated minute and generates fresh
telemetry, which is fed straight into the detection engine.

The important part for a live demo:

    cluster.inject_ghost(job_id, "nan_loss")

kills a job silently while everyone is watching — the GPU stays busy, nothing
crashes, and a few minutes later GhostGPU turns that row red on its own.
That is the moment worth showing on stage.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C
from engine import ENGINE_SETTINGS, get_engine

WINDOW = 30            # minutes of history the engine scores
MODEL_NAMES = ["unet3d-brats", "resnet50-chest", "vit-histo",
               "unet2d-lung", "effnet-derm", "swin-mri"]
# Background jobs use lab-style account names so they can never collide with a
# real person who types their own name on the researcher page.
OWNERS = ["lab-user-01", "lab-user-02", "lab-user-03",
          "lab-user-04", "lab-user-05", "lab-user-06"]


@dataclass
class LiveJob:
    job_id: int
    name: str
    owner: str
    gpu_id: int
    minute: int = 0
    # ghost state
    is_ghost: bool = False
    ghost_type: str = "none"
    ghost_onset: int = -1
    killed: bool = False
    # telemetry buffer (last ~90 minutes)
    rows: list = field(default_factory=list)
    # per-job personality so every job looks different
    base_gpu: float = 86.0
    amp: float = 6.0
    period: float = 3.0
    ckpt_every: int = 15
    loss_floor: float = 0.25
    loss_start: float = 2.4
    frozen_loss: float = 0.0
    # latest verdict from the engine
    verdict: dict = field(default_factory=dict)
    # False-Kill Guard: a kill must be confirmed by the job owner
    kill_requested: bool = False          # operator has asked to stop it
    kill_request_minute: int = -1
    evidence: dict = field(default_factory=dict)   # snapshot shown to the owner


class LiveCluster:
    def __init__(self, n_jobs: int = 8, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.engine = get_engine()
        self.jobs: dict[int, LiveJob] = {}
        self.killed_log: list[dict] = []      # what the owner sees afterwards
        self.next_id = 1
        self.total_reclaimed_cost = 0.0
        self.total_reclaimed_minutes = 0
        # False-Kill Guard tallies
        self.confirmed_kills = 0          # owner agreed the job was dead
        self.false_alarms = 0            # owner said the job was still alive
        self.decision_log: list[dict] = []
        for _ in range(n_jobs):
            self.add_job()
        # give every job some history so verdicts start immediately
        self.warmup(WINDOW + 5)

    # ------------------------------------------------------------- job set-up
    def add_job(self, owner: str | None = None,
                name: str | None = None) -> LiveJob:
        jid = self.next_id
        self.next_id += 1
        r = self.rng
        job = LiveJob(
            job_id=jid,
            name=name or random.choice(MODEL_NAMES),
            owner=owner or random.choice(OWNERS),
            gpu_id=int(r.integers(0, 8)),
            base_gpu=float(r.uniform(80, 92)),
            amp=float(r.uniform(3.5, 8.0)),
            period=float(r.uniform(2.0, 4.5)),
            ckpt_every=int(r.integers(10, 26)),
            loss_floor=float(r.uniform(0.15, 0.5)),
            loss_start=float(r.uniform(1.8, 3.0)),
        )
        self.jobs[jid] = job
        return job

    def submit_job(self, owner: str, name: str | None = None) -> dict:
        """A researcher starts a training run from their own screen."""
        owner = (owner or "").strip()[:24] or "anonymous"
        job = self.add_job(owner=owner, name=(name or "").strip()[:32] or None)
        # give it enough backfilled telemetry to be scored immediately
        for _ in range(WINDOW + 2):
            self._emit(job)
            self._score(job)
        return {"ok": True, "job_id": job.job_id, "name": job.name,
                "owner": job.owner}

    # --------------------------------------------------------------- ticking
    def _emit(self, job: LiveJob) -> dict:
        """Generate one minute of telemetry for a job."""
        r = self.rng
        m = job.minute

        if job.is_ghost and m >= job.ghost_onset:
            gt = job.ghost_type
            if gt == "nan_loss":
                # loop still runs: GPU rhythm stays normal, loss is NaN
                gpu = job.base_gpu + job.amp * np.sin(m / job.period) + r.normal(0, 2.0)
                loss = None
                ckpt = 0 if (m % job.ckpt_every) else int(r.random() > 0.5)
            elif gt == "frozen":
                gpu = r.uniform(87, 93) + r.normal(0, 0.8)
                loss = job.frozen_loss + float(r.normal(0, 0.0015))
                ckpt = 0
            else:  # stalled
                gpu = r.uniform(94, 98) + r.normal(0, 0.5)
                loss = job.frozen_loss + float(r.normal(0, 0.001))
                ckpt = 0
            mem = 70 + r.normal(0, 1.0)
        else:
            # healthy minute
            gpu = job.base_gpu + job.amp * np.sin(m / job.period) + r.normal(0, 2.5)
            if m % 37 < 3:                       # brief validation dip
                gpu -= r.uniform(25, 40)
            loss = float(job.loss_floor
                         + (job.loss_start - job.loss_floor) * np.exp(-m / 90.0)
                         + r.normal(0, 0.02))
            job.frozen_loss = loss               # remember for a future freeze
            mem = 70 + r.normal(0, 3)
            ckpt = int(m % job.ckpt_every == 0)

        row = dict(minute=m,
                   gpu_util=round(float(np.clip(gpu, 0, 100)), 2),
                   mem_util=round(float(np.clip(mem, 0, 100)), 2),
                   loss=(None if loss is None else round(float(loss), 4)),
                   checkpoint=int(ckpt))
        job.rows.append(row)
        job.rows = job.rows[-90:]
        job.minute += 1
        return row

    def _score(self, job: LiveJob):
        if len(job.rows) < WINDOW:
            job.verdict = dict(verdict="STARTING", ghost_prob=0.0,
                               reason="collecting telemetry",
                               wasted_cost=0.0, wasted_minutes=0)
            return
        win = pd.DataFrame(job.rows[-WINDOW:])
        job.verdict = self.engine.score(
            job.job_id, win, current_minute=job.minute,
            elapsed_min=float(job.minute))

    def tick(self):
        """Advance the whole cluster by one simulated minute."""
        for job in list(self.jobs.values()):
            if job.killed:
                continue
            self._emit(job)
            self._score(job)

    def warmup(self, minutes: int):
        for _ in range(minutes):
            self.tick()

    # ----------------------------------------------------------- live actions
    def inject_ghost(self, job_id: int, ghost_type: str = "nan_loss") -> dict:
        """Silently kill a running job — the live demo moment."""
        job = self.jobs.get(job_id)
        if job is None or job.killed:
            return {"ok": False, "error": "no such running job"}
        if ghost_type not in C.GHOST_TYPES:
            return {"ok": False, "error": f"unknown type; use {C.GHOST_TYPES}"}
        job.is_ghost = True
        job.ghost_type = ghost_type
        job.ghost_onset = job.minute
        return {"ok": True, "job_id": job_id, "ghost_type": ghost_type,
                "onset_minute": job.minute}

    # ---------------------- False-Kill Guard workflow -----------------------
    def build_evidence(self, job: LiveJob) -> dict:
        """
        The Evidence Card: the several signs, held over a time window, that
        justify calling this job a ghost. This is what the owner sees before
        deciding — no single quiet minute, but a picture over time.
        """
        rows = job.rows[-60:]
        loss_series, gpu_series = [], []
        for r in rows:
            loss_series.append(None if r["loss"] is None else round(r["loss"], 4))
            gpu_series.append(r["gpu_util"])

        v = job.verdict or {}
        hist = v.get("history", [])
        prob_series = [p for _, p in hist][-30:]

        # the multi-sign checklist (each is one independent piece of evidence)
        recent = rows[-WINDOW:] if len(rows) >= WINDOW else rows
        losses = [r["loss"] for r in recent]
        nan_share = sum(1 for l in losses if l is None) / max(len(losses), 1)
        real = [l for l in losses if l is not None]
        loss_move = (max(real) - min(real)) if real else 0.0
        gpus = [r["gpu_util"] for r in recent]
        gpu_mean = sum(gpus) / max(len(gpus), 1)
        gpu_spread = (max(gpus) - min(gpus)) if gpus else 0.0
        ck = [r["checkpoint"] for r in recent]
        since_ckpt = (len(ck) - 1 - max(i for i, c in enumerate(ck) if c)) if any(ck) else len(ck)

        signs = [
            {"label": "Loss stopped improving",
             "value": ("loss is NaN" if nan_share > 0.3
                       else f"moved only {loss_move:.4f} in {len(recent)} min"),
             "triggered": bool(nan_share > 0.3 or loss_move < 0.01)},
            {"label": "GPU busy but rhythm flat",
             "value": f"{gpu_mean:.0f}% used, only {gpu_spread:.1f}% variation",
             "triggered": bool(gpu_mean > 70 and gpu_spread < 6)},
            {"label": "No checkpoint written",
             "value": f"{since_ckpt} min since last checkpoint",
             "triggered": bool(since_ckpt >= 25)},
            {"label": "Sustained over time (not one blip)",
             "value": f"abnormal for {v.get('streak', 0)} windows in a row",
             "triggered": bool(v.get("streak", 0) >= self.engine.settings()["ghost_streak"])},
        ]
        n_trig = sum(1 for s in signs if s["triggered"])
        return dict(
            job_id=job.job_id, name=job.name, owner=job.owner,
            verdict=v.get("verdict", ""), ghost_prob=v.get("ghost_prob", 0.0),
            reason=v.get("reason", ""),
            signs=signs, signs_triggered=n_trig, signs_total=len(signs),
            loss_series=loss_series, gpu_series=gpu_series, prob_series=prob_series,
            wasted_minutes=v.get("wasted_minutes", 0),
            wasted_cost=v.get("wasted_cost", 0.0),
        )

    def request_kill(self, job_id: int) -> dict:
        """
        Operator asks to stop a job. This does NOT kill it — it sends the job
        owner a confirmation request with an evidence card. Nothing is stopped
        until the owner approves.
        """
        job = self.jobs.get(job_id)
        if job is None or job.killed:
            return {"ok": False, "error": "no such running job"}
        job.kill_requested = True
        job.kill_request_minute = job.minute
        job.evidence = self.build_evidence(job)
        return {"ok": True, "job_id": job_id, "awaiting": "owner confirmation",
                "evidence": job.evidence}

    def cancel_kill_request(self, job_id: int) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            return {"ok": False, "error": "no such job"}
        job.kill_requested = False
        return {"ok": True, "job_id": job_id}

    def owner_decision(self, job_id: int, approve: bool) -> dict:
        """
        The job owner answers the operator's request.
          approve=True  -> owner agrees it is dead: stop it, reclaim the GPU.
          approve=False -> owner says it is still alive: spare it, and count
                           this as a FALSE ALARM (the guard just prevented a
                           wrong kill).
        """
        job = self.jobs.get(job_id)
        if job is None or job.killed:
            return {"ok": False, "error": "no such running job"}
        if not job.kill_requested:
            return {"ok": False, "error": "no pending request for this job"}

        evidence = job.evidence or self.build_evidence(job)
        job.kill_requested = False

        if not approve:
            # owner rescued the job — a prevented false kill
            self.false_alarms += 1
            job.is_ghost = False          # treat as alive again
            job.ghost_onset = -1
            self.engine.reset(job_id)
            self.decision_log.append(dict(
                job_id=job.job_id, name=job.name, owner=job.owner,
                decision="spared", was_truly_ghost=evidence.get("verdict") == "GHOST",
                minute=job.minute))
            self.decision_log = self.decision_log[-50:]
            return {"ok": True, "decision": "spared", "job_id": job_id,
                    "false_alarm": True}

        # owner approved the kill
        wasted = job.verdict.get("wasted_minutes", 0)
        cost = job.verdict.get("wasted_cost", 0.0)
        job.killed = True
        self.confirmed_kills += 1
        self.killed_log.append(dict(
            job_id=job.job_id, name=job.name, owner=job.owner,
            reason=job.verdict.get("reason", ""),
            ghost_prob=job.verdict.get("ghost_prob", 0.0),
            wasted_minutes=wasted, wasted_cost=round(cost, 2),
            wasted_kwh=job.verdict.get("wasted_kwh", 0.0), minute=job.minute))
        self.killed_log = self.killed_log[-50:]
        self.decision_log.append(dict(
            job_id=job.job_id, name=job.name, owner=job.owner,
            decision="killed", was_truly_ghost=True, minute=job.minute))
        self.decision_log = self.decision_log[-50:]
        self.total_reclaimed_minutes += wasted
        self.total_reclaimed_cost += cost
        self.engine.reset(job_id)
        replacement = self.add_job()
        return {"ok": True, "decision": "killed", "killed": job_id,
                "reclaimed_minutes": wasted, "reclaimed_cost": round(cost, 2),
                "new_job_started": replacement.job_id}

    def kill_job(self, job_id: int) -> dict:
        """Backward-compatible direct kill (used only internally / tests)."""
        job = self.jobs.get(job_id)
        if job is None or job.killed:
            return {"ok": False, "error": "no such running job"}
        job.kill_requested = True
        job.evidence = self.build_evidence(job)
        return self.owner_decision(job_id, approve=True)

    # ------------------------------------------------------------- reporting
    def owner_view(self, owner: str) -> dict:
        """Everything one researcher should see about their own jobs."""
        owner = (owner or "").strip()
        mine = []
        for job in self.jobs.values():
            if job.killed or job.owner != owner:
                continue
            v = job.verdict or {}
            last = job.rows[-1] if job.rows else {}
            mine.append(dict(
                job_id=job.job_id, name=job.name, minute=job.minute,
                gpu_util=last.get("gpu_util"), loss=last.get("loss"),
                verdict=v.get("verdict", "STARTING"),
                ghost_prob=v.get("ghost_prob", 0.0),
                reason=v.get("reason", ""),
                wasted_cost=v.get("wasted_cost", 0.0),
                kill_requested=job.kill_requested,
                evidence=job.evidence if job.kill_requested else None,
                status="running"))
        killed = [k for k in self.killed_log if k["owner"] == owner][-5:]
        pending = [j for j in mine if j["kill_requested"]]
        return dict(owner=owner, jobs=mine, terminated=list(reversed(killed)),
                    pending=pending)

    def snapshot(self) -> dict:
        jobs = []
        for job in self.jobs.values():
            if job.killed:
                continue
            v = job.verdict or {}
            last = job.rows[-1] if job.rows else {}
            jobs.append(dict(
                job_id=job.job_id, name=job.name, owner=job.owner,
                gpu_id=job.gpu_id, minute=job.minute,
                gpu_util=last.get("gpu_util"), loss=last.get("loss"),
                verdict=v.get("verdict", "STARTING"),
                ghost_prob=v.get("ghost_prob", 0.0),
                reason=v.get("reason", ""),
                wasted_minutes=v.get("wasted_minutes", 0),
                wasted_cost=v.get("wasted_cost", 0.0),
                wasted_kwh=v.get("wasted_kwh", 0.0),
                spark=[p for _, p in v.get("history", [])][-24:],
                kill_requested=job.kill_requested,
                awaiting_owner=job.kill_requested,
                truth_ghost=job.is_ghost,          # for your own verification
                truth_type=job.ghost_type,
            ))
        jobs.sort(key=lambda j: (-j["ghost_prob"], j["job_id"]))

        ghosts = [j for j in jobs if j["verdict"] == "GHOST"]
        total_decided = self.confirmed_kills + self.false_alarms
        false_alarm_rate = (self.false_alarms / total_decided) if total_decided else 0.0
        return dict(
            jobs=jobs,
            stats=dict(
                running=len(jobs),
                ghosts=len(ghosts),
                burning_cost=round(sum(j["wasted_cost"] for j in ghosts), 2),
                burning_minutes=int(sum(j["wasted_minutes"] for j in ghosts)),
                reclaimed_cost=round(self.total_reclaimed_cost, 2),
                reclaimed_minutes=int(self.total_reclaimed_minutes),
                # False-Kill Guard tallies
                confirmed_kills=self.confirmed_kills,
                false_alarms=self.false_alarms,
                false_alarm_rate=round(false_alarm_rate, 3),
                awaiting_owner=sum(1 for j in jobs if j["kill_requested"]),
            ),
            decision_log=list(reversed(self.decision_log[-8:])),
            settings=dict(ENGINE_SETTINGS),
            ghost_types=C.GHOST_TYPES,
        )
