"""
GhostGPU — Step 6: the backend.

Runs the live cluster in the background and exposes it over a small API, plus
serves the dashboard itself.

Start it:
    uvicorn app:app --reload --app-dir src
then open  http://127.0.0.1:8000

Endpoints
    GET  /api/state              current jobs, verdicts, stats, settings
    POST /api/inject/{job_id}    silently kill a job  (?ghost_type=nan_loss)
    POST /api/kill/{job_id}      act on a verdict: stop job, reclaim the GPU
    POST /api/settings           change thresholds live (JSON body)
    POST /api/speed              change how fast simulated time runs
    POST /api/add_job            start another training job
    GET  /api/model              model info for the judges' questions
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

import config as C
from engine import get_engine
from live_cluster import LiveCluster

app = FastAPI(title="GhostGPU", version="1.0")

STATIC_DIR = C.ROOT / "static"
cluster = LiveCluster(n_jobs=8, seed=7)

# how many real seconds per simulated minute
TICK = {"seconds": 0.6, "paused": False}


async def ticker():
    while True:
        if not TICK["paused"]:
            cluster.tick()
        await asyncio.sleep(TICK["seconds"])


@app.on_event("startup")
async def start_ticker():
    asyncio.create_task(ticker())


# --------------------------------------------------------------------- pages
@app.get("/")
def landing():
    """Public landing page (about, how it works, team, contact)."""
    return FileResponse(STATIC_DIR / "landing.html")


@app.get("/dashboard")
def dashboard():
    """Operator / judge view — the full cluster."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/researcher")
def researcher_page():
    """Researcher view — a person sees only their own jobs."""
    return FileResponse(STATIC_DIR / "researcher.html")


# ----------------------------------------------------------------------- api
@app.get("/api/state")
def state():
    snap = cluster.snapshot()
    snap["tick"] = dict(TICK)
    return snap


@app.post("/api/inject/{job_id}")
def inject(job_id: int, ghost_type: str = "nan_loss"):
    return cluster.inject_ghost(job_id, ghost_type)


@app.post("/api/request_kill/{job_id}")
def request_kill(job_id: int):
    """Operator asks to stop a job -> sends the owner a confirmation request."""
    return cluster.request_kill(job_id)


@app.post("/api/cancel_request/{job_id}")
def cancel_request(job_id: int):
    return cluster.cancel_kill_request(job_id)


@app.post("/api/decision/{job_id}")
def owner_decision(job_id: int, approve: bool):
    """The job owner approves (kill) or denies (spare + false alarm)."""
    return cluster.owner_decision(job_id, approve)


@app.get("/api/evidence/{job_id}")
def evidence(job_id: int):
    job = cluster.jobs.get(job_id)
    if job is None:
        return {"ok": False, "error": "no such job"}
    return cluster.build_evidence(job)


@app.post("/api/kill/{job_id}")
def kill(job_id: int):
    """Legacy direct kill (kept for compatibility)."""
    return cluster.kill_job(job_id)


@app.post("/api/add_job")
def add_job():
    job = cluster.add_job()
    return {"ok": True, "job_id": job.job_id, "name": job.name}


@app.post("/api/submit")
def submit(owner: str, name: str = ""):
    """A researcher launches their own training job."""
    return cluster.submit_job(owner, name)


@app.get("/api/my")
def my_jobs(owner: str):
    """What one researcher sees about their own work."""
    return cluster.owner_view(owner)


@app.post("/api/settings")
async def settings(payload: dict):
    """Change detection thresholds while the system is running."""
    return {"ok": True, "settings": get_engine().update_settings(**payload)}


@app.post("/api/speed")
def speed(seconds: float | None = None, paused: bool | None = None):
    if seconds is not None:
        TICK["seconds"] = max(0.05, min(5.0, float(seconds)))
    if paused is not None:
        TICK["paused"] = bool(paused)
    return dict(TICK)


@app.get("/api/model")
def model_info():
    meta_path = C.ROOT / "models" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return JSONResponse({
        "stage_1": "Isolation Forest (unsupervised) — learns normal training, "
                   "flags anything unfamiliar, catches unseen failure types",
        "stage_2": "XGBoost (supervised) — turns signals + anomaly score into a "
                   "calibrated ghost probability",
        "hysteresis": f"a job must look anomalous for "
                      f"{get_engine().settings()['ghost_streak']} consecutive "
                      f"windows before it is called a GHOST",
        "metrics": meta,
    })
