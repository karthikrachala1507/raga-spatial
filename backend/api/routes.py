"""
Raga Spatial — API Routes
All endpoints the React frontend will call.
"""

import uuid
import os
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import aiofiles

router = APIRouter()

# ── In-memory job store (use Redis in production) ─────────────────────────
# Format: { job_id: { status, step, progress, result_path, error } }
jobs = {}


# ══════════════════════════════════════════════════════════════
# POST /api/upload
# Accept audio file, save it, return a job_id
# ══════════════════════════════════════════════════════════════
@router.post("/upload")
async def upload_song(file: UploadFile = File(...)):
    # Validate file type
    allowed = [".mp3", ".wav", ".flac", ".aac", ".ogg"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Use MP3, WAV, FLAC, AAC.")

    # Create unique job id
    job_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = f"outputs/uploads/{job_id}{ext}"
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Register job
    jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "step": "Waiting to start",
        "progress": 0,
        "upload_path": upload_path,
        "result": None,
        "error": None
    }

    return {
        "job_id": job_id,
        "filename": file.filename,
        "message": "Upload successful. Call /api/analyze/{job_id} to start."
    }


# ══════════════════════════════════════════════════════════════
# POST /api/analyze/{job_id}
# Trigger the full AI pipeline on the uploaded file
# ══════════════════════════════════════════════════════════════
@router.post("/analyze/{job_id}")
async def analyze_song(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        return {"message": "Already processing", "job_id": job_id}

    # Start pipeline in background
    jobs[job_id]["status"] = "processing"
    background_tasks.add_task(run_pipeline, job_id)

    return {
        "job_id": job_id,
        "message": "Analysis started. Poll /api/status/{job_id} for progress."
    }


# ══════════════════════════════════════════════════════════════
# GET /api/status/{job_id}
# Frontend polls this every 2s to show progress
# ══════════════════════════════════════════════════════════════
@router.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],        # uploaded | processing | complete | error
        "step": job["step"],            # human-readable current step
        "progress": job["progress"],    # 0–100
        "filename": job["filename"],
        "error": job["error"]
    }


# ══════════════════════════════════════════════════════════════
# GET /api/result/{job_id}
# Returns the full JSON metadata after analysis is complete
# ══════════════════════════════════════════════════════════════
@router.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"Job not complete yet. Status: {job['status']}")

    json_path = f"outputs/json/{job_id}.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Result JSON not found")

    with open(json_path, "r") as f:
        return JSONResponse(content=json.load(f))


# ══════════════════════════════════════════════════════════════
# GET /api/download/{job_id}
# Download the final output_spatial.wav
# ══════════════════════════════════════════════════════════════
@router.get("/download/{job_id}")
def download_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    wav_path = f"outputs/spatial/{job_id}_spatial.wav"
    if not os.path.exists(wav_path):
        raise HTTPException(status_code=404, detail="Spatial WAV not ready yet")

    return FileResponse(
        wav_path,
        media_type="audio/wav",
        filename=f"raga_spatial_{job_id[:8]}.wav"
    )


# ══════════════════════════════════════════════════════════════
# GET /api/jobs
# List all jobs (useful for debugging)
# ══════════════════════════════════════════════════════════════
@router.get("/jobs")
def list_jobs():
    return [
        {
            "job_id": j["job_id"],
            "filename": j["filename"],
            "status": j["status"],
            "progress": j["progress"]
        }
        for j in jobs.values()
    ]


# ══════════════════════════════════════════════════════════════
# PIPELINE RUNNER — called in background
# This will later call each core module in sequence
# ══════════════════════════════════════════════════════════════
def update_job(job_id: str, step: str, progress: int):
    """Helper to update job status."""
    jobs[job_id]["step"] = step
    jobs[job_id]["progress"] = progress
    print(f"[{job_id[:8]}] {progress}% — {step}")


def run_pipeline(job_id: str):
    """
    Orchestrates the full AI pipeline.
    Each step will call the real core module once built.
    For now, steps are stubbed so the API works end-to-end.
    """
    try:
        job = jobs[job_id]
        upload_path = job["upload_path"]

        # ── STEP 1: Preprocess ────────────────────────────────
        update_job(job_id, "Preprocessing audio...", 10)
        # from core.preprocess import preprocess_audio
        # audio, sr = preprocess_audio(upload_path)

        # ── STEP 2: Source Separation ─────────────────────────
        update_job(job_id, "BS-RoFormer: separating stems...", 25)
        # from core.separator import separate_stems
        # stems = separate_stems(audio, sr, job_id)

        # ── STEP 3: Feature Extraction ────────────────────────
        update_job(job_id, "Extracting mel spectrograms...", 40)
        # from core.preprocess import extract_features
        # features = extract_features(audio, sr)

        # ── STEP 4: BEATs Detection ───────────────────────────
        update_job(job_id, "BEATs: detecting instruments...", 55)
        # from core.detector import detect_instruments
        # detections = detect_instruments(audio, sr)

        # ── STEP 5: Motion Detection ──────────────────────────
        update_job(job_id, "Analyzing motion events...", 68)
        # from core.motion import detect_motion_events
        # motion_events = detect_motion_events(features, detections)

        # ── STEP 6: Spatial Assignment ────────────────────────
        update_job(job_id, "Assigning spatial positions...", 78)
        # from core.spatial import assign_directions
        # directions = assign_directions(detections, motion_events)

        # ── STEP 7: HRTF Convolution ──────────────────────────
        update_job(job_id, "SOFA HRTF: rendering binaural audio...", 88)
        # from core.hrtf import apply_hrtf
        # binaural_stems = apply_hrtf(stems, directions)

        # ── STEP 8: Final Render ──────────────────────────────
        update_job(job_id, "Rendering final spatial WAV...", 95)
        # from core.renderer import render_final
        # render_final(binaural_stems, job_id)

        # ── STEP 9: Export JSON ───────────────────────────────
        update_job(job_id, "Exporting metadata JSON...", 98)
        dummy_result = {
            "job_id": job_id,
            "filename": job["filename"],
            "duration": 0,
            "instruments": [],
            "motion_events": [],
            "spatial_assignments": {},
            "note": "Pipeline stubs active — real modules coming in Phase 3+"
        }
        json_path = f"outputs/json/{job_id}.json"
        with open(json_path, "w") as f:
            json.dump(dummy_result, f, indent=2)

        # ── DONE ──────────────────────────────────────────────
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["step"] = "Complete"
        jobs[job_id]["result"] = json_path
        print(f"[{job_id[:8]}] Pipeline complete.")

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["step"] = f"Error: {e}"
        print(f"[{job_id[:8]}] ERROR: {e}")
