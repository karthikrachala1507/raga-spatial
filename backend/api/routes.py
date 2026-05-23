# Raga Spatial - api/routes.py
# Phase 10: Full pipeline connected to FastAPI endpoints

import os
import uuid
import time
import json
import shutil
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "outputs", "uploads")
STEMS_DIR   = os.path.join(BASE_DIR, "outputs", "stems")
SPATIAL_DIR = os.path.join(BASE_DIR, "outputs", "spatial")
JSON_DIR    = os.path.join(BASE_DIR, "outputs", "json")

for d in [UPLOAD_DIR, STEMS_DIR, SPATIAL_DIR, JSON_DIR]:
    os.makedirs(d, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { "status", "step", "message", "result", "error", "created_at" } }
JOBS = {}
JOBS_LOCK = threading.Lock()


def _update_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _run_pipeline_thread(job_id, audio_path):
    """Run full pipeline in background thread."""
    try:
        _update_job(job_id, status="running", step=1,
                    message="Starting pipeline...")

        def progress(step, msg):
            _update_job(job_id, step=step, message=msg)

        from core.renderer import run_full_pipeline
        result = run_full_pipeline(
            audio_path        = audio_path,
            job_id            = job_id,
            progress_callback = progress,
        )

        _update_job(job_id,
                    status  = "complete",
                    step    = 7,
                    message = "Pipeline complete",
                    result  = result)

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print("[Routes] Pipeline error for job " + job_id + ":\n" + err)
        _update_job(job_id,
                    status  = "error",
                    message = str(e),
                    error   = err)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_song(file: UploadFile = File(...)):
    """
    Upload a song and start the full Raga Spatial pipeline.

    Returns job_id immediately. Poll /status/{job_id} for progress.
    Supported formats: mp3, wav, flac, m4a, ogg
    """
    allowed = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
    ext     = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400,
            detail="Unsupported format. Use: " + ", ".join(allowed))

    job_id    = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, job_id + ext)

    # Save uploaded file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    file_mb = round(os.path.getsize(save_path) / (1024*1024), 2)
    print("[Routes] Upload: " + file.filename
          + " -> " + job_id + " (" + str(file_mb) + " MB)")

    # Register job
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id":     job_id,
            "filename":   file.filename,
            "audio_path": save_path,
            "status":     "queued",
            "step":       0,
            "message":    "Queued",
            "result":     None,
            "error":      None,
            "created_at": time.time(),
        }

    # Start pipeline in background thread
    thread = threading.Thread(
        target = _run_pipeline_thread,
        args   = (job_id, save_path),
        daemon = True,
    )
    thread.start()

    return JSONResponse({
        "job_id":    job_id,
        "filename":  file.filename,
        "status":    "queued",
        "message":   "Pipeline started. Poll /status/" + job_id,
        "endpoints": {
            "status":   "/status/"   + job_id,
            "result":   "/result/"   + job_id,
            "download": "/download/" + job_id,
        }
    })


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get pipeline status for a job.

    Returns:
        status   : "queued" | "running" | "complete" | "error"
        step     : 0-7 (current pipeline step)
        message  : human-readable progress message
        progress : 0-100 percentage
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        raise HTTPException(status_code=404,
            detail="Job not found: " + job_id)

    progress_pct = round(job["step"] / 7 * 100)

    return JSONResponse({
        "job_id":   job_id,
        "status":   job["status"],
        "step":     job["step"],
        "message":  job["message"],
        "progress": progress_pct,
        "filename": job.get("filename", ""),
    })


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Get full pipeline result for a completed job.
    Includes detection summary, motion events, spatial timeline.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        raise HTTPException(status_code=404,
            detail="Job not found: " + job_id)

    if job["status"] == "running" or job["status"] == "queued":
        return JSONResponse({
            "job_id":  job_id,
            "status":  job["status"],
            "message": "Pipeline still running. Step "
                       + str(job["step"]) + "/7: " + job["message"],
        }, status_code=202)

    if job["status"] == "error":
        return JSONResponse({
            "job_id":  job_id,
            "status":  "error",
            "message": job["message"],
        }, status_code=500)

    result = job.get("result", {})

    # Load JSON output file if available
    json_path = result.get("json_path") if result else None
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            full_result = json.load(f)
        return JSONResponse(full_result)

    # Fallback: return in-memory result
    return JSONResponse({
        "job_id":            job_id,
        "status":            "complete",
        "duration_sec":      result.get("duration_sec"),
        "tempo_bpm":         result.get("tempo_bpm"),
        "detection_summary": result.get("detection_summary", []),
        "motion_summary":    result.get("motion_summary", {}),
        "motion_events":     result.get("motion_events", []),
        "spatial_wav":       result.get("spatial_wav"),
    })


@router.get("/download/{job_id}")
async def download_spatial(job_id: str):
    """
    Download the spatial WAV output for a completed job.
    This is the binaural audio — listen with headphones.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        raise HTTPException(status_code=404,
            detail="Job not found: " + job_id)

    if job["status"] != "complete":
        raise HTTPException(status_code=409,
            detail="Job not complete yet. Status: " + job["status"])

    result    = job.get("result", {})
    wav_path  = result.get("spatial_wav") if result else None

    if not wav_path or not os.path.exists(wav_path):
        # Try to find it by convention
        wav_path = os.path.join(SPATIAL_DIR, job_id + "_spatial.wav")

    if not os.path.exists(wav_path):
        raise HTTPException(status_code=404,
            detail="Spatial WAV not found for job: " + job_id)

    return FileResponse(
        path             = wav_path,
        media_type       = "audio/wav",
        filename         = job_id + "_spatial.wav",
        headers          = {"Content-Disposition":
                            "attachment; filename=" + job_id + "_spatial.wav"}
    )


@router.get("/download/{job_id}/stems/{stem_name}")
async def download_stem(job_id: str, stem_name: str):
    """
    Download a separated stem (vocals or instrumental).
    stem_name: "vocals" or "instrumental"
    """
    if stem_name not in ("vocals", "instrumental"):
        raise HTTPException(status_code=400,
            detail="stem_name must be 'vocals' or 'instrumental'")

    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if job is None:
        raise HTTPException(status_code=404,
            detail="Job not found: " + job_id)

    if job["status"] != "complete":
        raise HTTPException(status_code=409,
            detail="Job not complete yet. Status: " + job["status"])

    stem_path = os.path.join(STEMS_DIR, job_id + "_" + stem_name + ".wav")
    if not os.path.exists(stem_path):
        raise HTTPException(status_code=404,
            detail="Stem not found: " + stem_name)

    return FileResponse(
        path       = stem_path,
        media_type = "audio/wav",
        filename   = job_id + "_" + stem_name + ".wav",
    )


@router.get("/jobs")
async def list_jobs():
    """List all jobs with their current status."""
    with JOBS_LOCK:
        jobs_list = [
            {
                "job_id":   jid,
                "filename": j.get("filename", ""),
                "status":   j["status"],
                "step":     j["step"],
                "message":  j["message"],
                "created_at": j.get("created_at", 0),
            }
            for jid, j in JOBS.items()
        ]

    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
    return JSONResponse({"jobs": jobs_list, "total": len(jobs_list)})


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and clean up its output files."""
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)

    if job is None:
        raise HTTPException(status_code=404,
            detail="Job not found: " + job_id)

    # Clean up files
    files_deleted = []
    for path in [
        job.get("audio_path"),
        os.path.join(SPATIAL_DIR, job_id + "_spatial.wav"),
        os.path.join(STEMS_DIR,   job_id + "_vocals.wav"),
        os.path.join(STEMS_DIR,   job_id + "_instrumental.wav"),
        os.path.join(JSON_DIR,    job_id + "_output.json"),
    ]:
        if path and os.path.exists(path):
            os.remove(path)
            files_deleted.append(path)

    return JSONResponse({
        "job_id":        job_id,
        "deleted":       True,
        "files_deleted": len(files_deleted),
    })


@router.get("/health")
async def health_check():
    """Health check — confirms server is running and models are available."""
    from core.renderer import get_pipeline_status
    status = get_pipeline_status()
    all_ok = all(v == "ok" for v in status.values())
    return JSONResponse({
        "status":          "ok" if all_ok else "degraded",
        "pipeline_status": status,
        "active_jobs":     sum(1 for j in JOBS.values()
                               if j["status"] == "running"),
        "total_jobs":      len(JOBS),
    })
