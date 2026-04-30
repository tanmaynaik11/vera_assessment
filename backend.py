import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import logging_config
logging_config.setup()

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vera Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = Path("runs")

# In-memory job store. Single-worker only — fine for local/dev use.
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, records: list[dict]):
    import pipeline as pl

    total = len(records)
    results = []

    for i, record in enumerate(records):
        query_preview = record.get("input", {}).get("question", "")[:60]

        def on_status(msg, _i=i, _q=query_preview):
            with _lock:
                _jobs[job_id]["current_step"] = msg
                _jobs[job_id]["current_query"] = _q

        with _lock:
            _jobs[job_id]["progress"] = i
            _jobs[job_id]["current_query"] = query_preview
            _jobs[job_id]["current_step"] = "starting..."

        try:
            result = pl.run_record(record, on_status=on_status)
        except Exception as exc:
            result = {
                "id": record.get("id", "?"),
                "query": record.get("input", {}).get("question", ""),
                "query_type": "ERROR",
                "verdict": "error",
                "verdict_text": str(exc),
                "error": str(exc),
                "claims": [],
                "information_gaps": [],
            }

        results.append(result)

    # Persist to runs/
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = RUNS_DIR / f"run_{timestamp}.json"
    run_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with _lock:
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["progress"] = total
        _jobs[job_id]["current_step"] = "complete"
        _jobs[job_id]["result"] = results
        _jobs[job_id]["run_file"] = run_file.name


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
async def start_evaluation(file: UploadFile):
    raw = await file.read()
    try:
        records = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse JSON body")

    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="JSON must be a top-level list of records")
    if len(records) == 0:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "total": len(records),
            "current_query": "",
            "current_step": "queued",
            "result": None,
            "run_file": None,
        }

    thread = threading.Thread(target=_run_job, args=(job_id, records), daemon=True)
    thread.start()

    return {"job_id": job_id, "total": len(records)}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "current_query": job["current_query"],
        "current_step": job["current_step"],
        "run_file": job["run_file"],
    }


@app.get("/result/{job_id}")
def get_result(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not finished yet")
    return job["result"]


@app.get("/runs")
def list_runs():
    if not RUNS_DIR.exists():
        return []
    return [p.name for p in sorted(RUNS_DIR.glob("run_*.json"), reverse=True)]


@app.get("/runs/{filename}")
def load_run(filename: str):
    run_file = RUNS_DIR / filename
    if not run_file.exists():
        raise HTTPException(status_code=404, detail="Run file not found")
    return json.loads(run_file.read_text(encoding="utf-8"))
