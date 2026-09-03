#!/usr/bin/env python3
"""
Local Wan 2.2 image-to-video server.

Deliberately mirrors the RunPod serverless API shape (/run, /runsync,
/status/{id}) so the OpenClaw MCP server can point at either backend with only
a base URL change.

Privacy posture: binds to 127.0.0.1 by default, so nothing is reachable off the
machine unless you change WAN_HOST on purpose. Videos are written to local disk
and served from this process — no object storage, no third-party queue, no
job payloads leaving the box.

    pip install -r local/requirements.txt
    python local/server.py
"""

import os
import sys
import threading
import queue
import uuid
import time
from datetime import datetime, timezone

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import loras
import wan_core

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
VRAM_MODE = os.environ.get("VRAM_MODE", "auto")

HOST = os.environ.get("WAN_HOST", "127.0.0.1")
PORT = int(os.environ.get("WAN_PORT", "8080"))
API_KEY = os.environ.get("WAN_API_KEY") or None

OUTPUT_DIR = os.path.abspath(os.environ.get("OUTPUT_DIR", "./outputs"))
PUBLIC_BASE = os.environ.get("WAN_PUBLIC_BASE", f"http://{HOST}:{PORT}")
# 0 keeps videos forever; otherwise a sweep deletes them after this many hours.
RETAIN_HOURS = float(os.environ.get("RETAIN_HOURS", "0"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="wan2.2-i2v-local", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# Job state
#
# One GPU means one job at a time, so a single worker thread drains a FIFO
# queue. Job records live in memory only — restarting the server forgets
# history, which is the desired behaviour for a private box.
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_pipe = None
_pipe_error: str | None = None


def _set(job_id: str, **fields) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _sweep_old_videos() -> None:
    """Delete generated videos older than RETAIN_HOURS."""
    if RETAIN_HOURS <= 0:
        return
    cutoff = time.time() - RETAIN_HOURS * 3600
    for name in os.listdir(OUTPUT_DIR):
        if not name.endswith(".mp4"):
            continue
        path = os.path.join(OUTPUT_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.unlink(path)
        except OSError:
            pass


def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    job_input = job["input"]

    try:
        params = wan_core.validate_params(job_input)
    except ValueError as exc:
        _set(job_id, status="FAILED", output={"error": str(exc)})
        return

    try:
        requested = loras.parse_request(job_input.get("loras"))
        applied = loras.apply(_pipe, requested)
    except loras.LoraError as exc:
        _set(job_id, status="FAILED", output={"error": str(exc)})
        return

    try:
        image = wan_core.load_image(job_input.get("image"), job_input.get("image_url"))
    except Exception as exc:
        _set(job_id, status="FAILED", output={"error": f"Failed to load image: {exc}"})
        return

    width, height = wan_core.calculate_dimensions(_pipe, image, params["resolution"])
    image = image.resize((width, height))
    params["width"], params["height"] = width, height

    _set(job_id, status="IN_PROGRESS",
         progress=f"Generating {params['resolution']} ({width}x{height}, "
                  f"{params['num_frames']} frames, {params['num_inference_steps']} steps)")

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if params["seed"] is not None:
        generator.manual_seed(int(params["seed"]))

    try:
        frames = wan_core.generate(_pipe, image, params, generator=generator)
    except Exception as exc:
        _set(job_id, status="FAILED", output={"error": f"Generation failed: {exc}"})
        return

    _set(job_id, status="IN_PROGRESS", progress="Encoding video")
    name = f"{job_id}.mp4"
    path = os.path.join(OUTPUT_DIR, name)
    try:
        wan_core.write_video(frames, path, fps=params["fps"])
    except Exception as exc:
        _set(job_id, status="FAILED", output={"error": f"Failed to write video: {exc}"})
        return

    _sweep_old_videos()
    _set(job_id, status="COMPLETED", progress=None, output={
        "video_url": f"{PUBLIC_BASE.rstrip('/')}/videos/{name}",
        "video_path": path,
        "width": width,
        "height": height,
        "num_frames": params["num_frames"],
        "fps": params["fps"],
        "resolution": params["resolution"],
        "loras": applied,
        "seed": params["seed"],
    })


def _worker() -> None:
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        except Exception as exc:  # never let the worker thread die
            _set(job_id, status="FAILED", output={"error": f"Unhandled error: {exc}"})
        finally:
            _queue.task_done()


def _load_model() -> None:
    """Load weights in the background so /health answers immediately."""
    global _pipe, _pipe_error
    try:
        _pipe = wan_core.load_pipeline(MODEL_ID, HF_TOKEN, VRAM_MODE)
    except Exception as exc:
        _pipe_error = str(exc)
        print(f"[wan] MODEL LOAD FAILED: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.middleware("http")
async def require_key(request: Request, call_next):
    if API_KEY and request.url.path != "/health":
        header = request.headers.get("authorization", "")
        if header != f"Bearer {API_KEY}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _pipe is not None,
        "model_error": _pipe_error,
        "vram_gb": round(wan_core.detect_vram_gb(), 1),
        "vram_mode": wan_core.resolve_vram_mode(VRAM_MODE),
        "queued": _queue.qsize(),
    }


def _submit(body: dict) -> str:
    job_input = (body or {}).get("input") or {}
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "input": job_input,
            "status": "IN_QUEUE",
            "created": datetime.now(timezone.utc).isoformat(),
            "output": None,
        }
    _queue.put(job_id)
    return job_id


@app.post("/run")
async def run(request: Request):
    body = await request.json()
    job_input = (body or {}).get("input") or {}

    if job_input.get("action") == "list_loras":
        return {"id": None, "status": "COMPLETED", "output": {"loras": loras.available()}}

    if _pipe is None:
        raise HTTPException(503, _pipe_error or "Model still loading; try again shortly.")

    job_id = _submit(body)
    return {"id": job_id, "status": "IN_QUEUE"}


@app.post("/runsync")
async def runsync(request: Request):
    body = await request.json()
    job_input = (body or {}).get("input") or {}

    # Cheap introspection needs no model and no queue.
    if job_input.get("action") == "list_loras":
        return {"status": "COMPLETED", "output": {"loras": loras.available()}}

    if _pipe is None:
        raise HTTPException(503, _pipe_error or "Model still loading; try again shortly.")

    job_id = _submit(body)
    while _jobs[job_id]["status"] in ("IN_QUEUE", "IN_PROGRESS"):
        time.sleep(1)
    job = _jobs[job_id]
    return {"id": job_id, "status": job["status"], "output": job["output"]}


@app.get("/status/{job_id}")
def status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return {
        "id": job_id,
        "status": job["status"],
        "output": job["output"],
        "progress": job.get("progress"),
    }


@app.get("/videos/{name}")
def video(name: str):
    # Serve only plain filenames from OUTPUT_DIR — never a traversal.
    if os.path.basename(name) != name or not name.endswith(".mp4"):
        raise HTTPException(400, "Invalid video name")
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "No such video")
    return FileResponse(path, media_type="video/mp4", filename=name)


def main() -> None:
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_load_model, daemon=True).start()
    if HOST not in ("127.0.0.1", "localhost", "::1") and not API_KEY:
        print(
            f"[wan] WARNING: binding to {HOST} without WAN_API_KEY set — "
            f"anyone who can reach this port can generate video.",
            file=sys.stderr,
        )
    import uvicorn

    print(f"[wan] Serving on http://{HOST}:{PORT} (outputs: {OUTPUT_DIR})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
