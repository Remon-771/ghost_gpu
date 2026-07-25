---
title: GhostGPU
emoji: 👻
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Live detection of GPU jobs that look busy but have silently died
---

# GhostGPU — live silent-failure detection for GPU clusters

> **Busy is not productive.**
> A training job can die silently — loss goes NaN, the data loader hangs, the
> process stalls — while the GPU still reports 95%+ utilisation. Every dashboard
> says the job is fine. It is burning money and blocking the queue for nothing.
> GhostGPU catches it.

---

## What's inside

| Piece | File | What it does |
|---|---|---|
| Data generator | `src/generate_data.py` | Simulates a realistic cluster: healthy + ghost jobs |
| Features | `src/features.py` | Turns raw telemetry into rolling-window signals |
| AI core | `src/train_model.py` | Trains Isolation Forest + XGBoost |
| Engine | `src/engine.py` | Verdicts, hysteresis, explanations, cost accounting |
| Live cluster | `src/live_cluster.py` | Jobs running in real time + **ghost injection** |
| Backend | `src/app.py` | FastAPI: state, inject, kill, live settings |
| Dashboard | `static/index.html` | The live screen you demo |

---

## Setup (once)

Open this folder in VS Code -> **Terminal -> New Terminal**:

```bash
python -m venv venv

# activate:
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

The venv is active when the prompt starts with `(venv)`.

> **Windows "running scripts is disabled"?** Run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Build the model (run once, in order)

```bash
python src/generate_data.py     # 1. simulate a cluster
python src/features.py          # 2. extract window features
python src/train_model.py       # 3. train the two-stage model
```

Expected from step 3 (roughly):

```
accuracy   : 0.991
ROC-AUC    : 1.000
recall per ghost type:  nan_loss 1.000 | stalled 1.000 | frozen 0.625
```

`frozen` is the hard case on purpose — a hung data loader looks almost exactly
like a long evaluation phase. Being honest about that is a strength in Q&A.

---

## Run the live dashboard

```bash
uvicorn app:app --reload --app-dir src
```

Open **http://127.0.0.1:8000**

You'll see 8 jobs training live, with GPU utilisation, loss, ghost risk and a
trend sparkline updating every second.

---

## The demo (this is the moment)

GhostGPU has **two screens**, which is what makes the demo land:

| Screen | URL | Who looks at it |
|---|---|---|
| **Operator / judge view** | `http://127.0.0.1:8000/` | The whole cluster, every job, verdicts, cost |
| **Researcher view** | `http://127.0.0.1:8000/researcher` | One person, their own jobs only |

### The two-screen story (recommended)

1. **A colleague opens `/researcher` on their own laptop or phone**, types their
   name, and starts a training job. It appears instantly on the judges' screen.
2. Everyone watches it train normally — loss falling, GPU busy, all green.
3. **Silently kill it** from the operator screen ("Inject silent failure").
   Point out: *the GPU is still busy, nothing crashed, no error anywhere.*
4. The researcher's own screen turns amber — **"GhostGPU is checking your job"** —
   then red: **"Your job has stopped learning."** They did not report it; the system found it.
5. The operator presses **Kill & free**. On the colleague's screen the job
   disappears and is replaced by: *"Your job was stopped by the cluster operator…
   it had already used N GPU-minutes without producing any result."*

That last moment — a real person's screen changing because the AI made a decision —
is far more convincing than any slide.

> For a colleague on another device, start the server with
> `python -m uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir src`
> and share `http://<your-ip>:8000/researcher` (find your IP with `ipconfig`).

### Solo version

1. Pick a job in **"Inject ghost into"**, choose a failure type, hit
   **Inject silent failure**.
2. Point out: *the GPU stays busy, nothing crashes, no error anywhere.*
3. Within ~30 simulated minutes the row turns **red**, the verdict flips to
   **GHOST**, and a live counter shows the money and kWh being burned.
4. Click **Kill & free** — the GPU is reclaimed and the savings move to the
   "Reclaimed" card.

Three ghost types to demo:

| Type | What broke | How it looks |
|---|---|---|
| `nan_loss` | Gradient exploded, loss is NaN | GPU rhythm perfectly normal, loss shows **NaN** |
| `frozen` | Data loader hung | Loss frozen, GPU busy but flat |
| `stalled` | Process deadlocked | GPU pinned ~96%, nothing moves |

---

## Adapting live (for an on-site twist)

Everything tunable is editable **while it runs**:

- **Dashboard controls** — ghost threshold, streak length, speed, pause, add jobs.
- **`POST /api/settings`** — `{"ghost_prob":0.5,"ghost_streak":3}`
- **New failure type** — add a name to `GHOST_TYPES` in `src/config.py`, then a
  matching branch in `generate_data.apply_ghost()` and `live_cluster._emit()`.
  It appears in the dashboard dropdown automatically.
- **Cost model** — `GPU_COST_PER_HOUR` in `src/config.py`.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/state` | Jobs, verdicts, stats, settings |
| `POST /api/inject/{job_id}?ghost_type=nan_loss` | Silently kill a job |
| `POST /api/kill/{job_id}` | Act on a verdict, reclaim the GPU |
| `POST /api/settings` | Change thresholds live |
| `POST /api/speed?seconds=0.4` / `?paused=true` | Simulation speed |
| `POST /api/add_job` | Start another job |
| `GET /api/model` | Model description + metrics (good for judges) |

---

## How the detection works

**Signals** (per 30-minute rolling window)
- *Learning progress* — loss slope, change, variance, NaN ratio
- *Utilisation rhythm* — mean, variance, range, flatness
- *Output activity* — checkpoint count, minutes since last checkpoint

**Two stages**
1. **Isolation Forest** (unsupervised) — trained only on healthy windows, so it
   flags failure patterns it has never seen before.
2. **XGBoost** (supervised) — combines the signals plus the anomaly score into a
   calibrated ghost probability.

**Hysteresis** — a job must look anomalous for several consecutive windows
before it is declared a GHOST, so benign quiet periods don't raise false alarms.

**Measured on held-out synthetic jobs, pooled over 5 random job-level splits.**
We report two failure-timing regimes, because quoting only the easy one overstates
what the system can do:

| Failure timing | Ghost jobs detected | False GHOST on healthy jobs | Latency |
|---|---|---|---|
| **Hard** — a job may die shortly before it would have finished *(default)* | **75.3%** (CI 64.9–84.4) | **1.3%** (CI 0.0–3.1) | 37 min |
| **Favourable** — every failure has ≥60 min of runtime left | **90.1%** (CI 84.0–96.3) | 1.8% (CI 0.5–3.7) | 37 min |

Switch regimes with `MIN_GHOST_RUNTIME` in `src/config.py` (0 = hard, 60 = favourable).

### Why machine learning is needed

| Detector | Job recall | False alarm |
|---|---|---|
| GPU-utilisation threshold | **0.0%** | 0.0% |
| Utilisation-flatness rule | 44.7% | 30.0% |
| Hand-written multi-signal rule | 70.3% | **48.5%** |
| Isolation Forest only | 18.7% | 19.2% |
| XGBoost only | 77.7% | 2.3% |
| **GhostGPU (two-stage)** | **76.4%** | **1.3%** |

The classic rule catches nothing; the best hand-written rule condemns half of all
healthy jobs. Run `python src/evaluate.py` to reproduce, including
leave-one-failure-type-out and operational-noise measurement.

### Calibration and methodology
- **Three-way split by job** — train (60%) / calibration (20%) / test (20%), fully
  disjoint. The test split is touched once.
- **Platt calibration** fitted on the calibration split. Test **Brier score 0.0043**,
  **expected calibration error 0.0152** — so "probability" is a defensible word.
- **Operating point** (threshold + streak) selected on the calibration split, never
  on test.
- **Bootstrap 95% confidence intervals** on every headline rate.

### Honest notes
1. All results are on *synthetic* failure injection. The public Philly trace has no
   silent-failure labels, so it motivates the problem but does not validate the detector.
2. On known failure types the two-stage model matches XGBoost alone; Stage 1 is kept
   because it needs no labels, not because we measured a gain.
3. Leave-one-failure-type-out shows the model does **not** yet generalise to unseen
   failure modes — this is our main open problem, and we report it rather than assume it.

## Why a simple rule doesn't work

"Alert if GPU utilisation drops" fails both ways: ghost jobs often keep the GPU
**busy**, and healthy jobs go quiet during validation. Telling them apart needs
several weak signals read together — which is exactly what the model does.
