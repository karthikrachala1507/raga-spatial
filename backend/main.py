"""
Raga Spatial — FastAPI Backend Entry Point
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from api.routes import router

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Raga Spatial API",
    description="AI-powered spatial music intelligence for South Indian cinematic music",
    version="1.0.0"
)

# ── CORS (allow React frontend to call this API) ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Output folders (create if missing) ────────────────────────────────────
os.makedirs("outputs/uploads", exist_ok=True)
os.makedirs("outputs/stems", exist_ok=True)
os.makedirs("outputs/spatial", exist_ok=True)
os.makedirs("outputs/json", exist_ok=True)

# ── Serve output files statically ─────────────────────────────────────────
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ── Register API routes ────────────────────────────────────────────────────
app.include_router(router, prefix="/api")


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status": "running",
        "service": "Raga Spatial API",
        "version": "1.0.0"
    }
