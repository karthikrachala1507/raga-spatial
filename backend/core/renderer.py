# Raga Spatial - core/renderer.py
# Phase 9: Final Renderer
#
# Connects all phases into one pipeline call:
# preprocess -> separate -> detect -> motion -> spatial -> hrtf
# Outputs: output_spatial.wav + output.json

import os
import json
import time
import numpy as np

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
STEMS_DIR   = os.path.join(OUTPUTS_DIR, "stems")
SPATIAL_DIR = os.path.join(OUTPUTS_DIR, "spatial")
JSON_DIR    = os.path.join(OUTPUTS_DIR, "json")

# Fixed stem-to-instrument mapping
# vocals  -> Singing (front center)
# instrumental -> Plucked string instrument (front left)
STEM_LABEL_MAP = {
    "vocals":       "Singing",
    "instrumental": "Plucked string instrument",
}


def run_full_pipeline(audio_path, job_id, progress_callback=None):
    """
    Run the complete Raga Spatial pipeline on a song.

    Args:
        audio_path        : path to input mp3/wav/flac
        job_id            : unique job identifier
        progress_callback : optional function(step, message) for UI updates

    Returns:
        dict with all outputs:
        {
            "job_id"          : str,
            "duration_sec"    : float,
            "spatial_wav"     : str,   path to output_spatial.wav
            "json_path"       : str,   path to output.json
            "stems"           : dict,  paths to vocals.wav + instrumental.wav
            "detections"      : list,  per-window instrument detections
            "motion_events"   : list,  motion events with timestamps
            "spatial_timeline": list,  per-second spatial positions
            "summary"         : dict,  high-level analysis summary
            "processing_time" : float, total seconds taken
        }
    """
    pipeline_start = time.time()
    os.makedirs(STEMS_DIR,   exist_ok=True)
    os.makedirs(SPATIAL_DIR, exist_ok=True)
    os.makedirs(JSON_DIR,    exist_ok=True)

    def _progress(step, msg):
        print("[Pipeline] [" + str(step) + "/7] " + msg)
        if progress_callback:
            progress_callback(step, msg)

    # Step 1: Preprocess
    _progress(1, "Preprocessing audio...")
    from core.preprocess import (load_audio, normalize_audio,
                                  extract_chroma, detect_onsets,
                                  get_tempo_and_beats, resample_for_beats)
    t = time.time()
    audio, sr     = load_audio(audio_path)
    audio         = normalize_audio(audio)
    duration_sec  = len(audio) / sr
    chroma        = extract_chroma(audio, sr)
    onsets        = detect_onsets(audio, sr)
    tempo, beats  = get_tempo_and_beats(audio, sr)
    audio_16k     = resample_for_beats(audio, sr)
    _progress(1, "Preprocessing done ("
              + str(round(duration_sec, 1)) + "s audio, "
              + str(len(onsets)) + " onsets, "
              + str(round(float(tempo), 0)) + " BPM) ["
              + str(round(time.time()-t, 1)) + "s]")

    # Step 2: Source separation
    _progress(2, "Separating stems (MelBandRoformer)...")
    t = time.time()
    from core.separator import separate_stems
    stems = separate_stems(audio, sr, job_id)
    _progress(2, "Separation done: "
              + str(list(stems.keys())) + " ["
              + str(round(time.time()-t, 1)) + "s]")

    # Step 3: Instrument detection
    _progress(3, "Detecting instruments (BEATs)...")
    t = time.time()
    from core.detector import detect_from_stems, summarise_detections
    det_results = detect_from_stems(stems["vocals"], stems["instrumental"])
    det_summary = summarise_detections(det_results, top_n=10)
    detections  = det_results["merged"]
    _progress(3, "Detection done: "
              + str(len(det_summary)) + " instrument classes ["
              + str(round(time.time()-t, 1)) + "s]")

    # Step 4: Motion detection
    _progress(4, "Detecting motion events...")
    t = time.time()
    from core.motion import detect_motion_events, build_motion_timeline, summarise_motion
    motion_events   = detect_motion_events(
        chroma     = chroma,
        onsets     = onsets,
        beats      = beats,
        detections = detections,
        sr         = sr,
        hop_length = 512,
    )
    motion_timeline = build_motion_timeline(motion_events, duration_sec)
    motion_summary  = summarise_motion(motion_events, duration_sec)
    _progress(4, "Motion done: "
              + str(len(motion_events)) + " events, "
              + str(motion_summary["motion_coverage_pct"]) + "% coverage ["
              + str(round(time.time()-t, 1)) + "s]")

    # Step 5: Spatial assignment
    _progress(5, "Assigning spatial positions...")
    t = time.time()
    from core.spatial import assign_spatial_positions
    spatial_result = assign_spatial_positions(
        detections         = detections,
        motion_events      = motion_events,
        total_duration_sec = duration_sec,
        resolution_sec     = 0.1,
    )
    # Attach fixed stem label map so HRTF renderer uses correct positions
    spatial_result["stem_labels"] = STEM_LABEL_MAP
    _progress(5, "Spatial done: "
              + str(len(spatial_result["timeline"])) + " time steps ["
              + str(round(time.time()-t, 1)) + "s]")

    # Step 6: HRTF binaural rendering
    _progress(6, "Rendering binaural audio (HRTF)...")
    t = time.time()
    from core.hrtf import render_binaural
    spatial_wav = render_binaural(
        stems_dir      = STEMS_DIR,
        job_id         = job_id,
        spatial_result = spatial_result,
        sr             = sr,
    )
    _progress(6, "HRTF render done ["
              + str(round(time.time()-t, 1)) + "s]")

    # Step 7: Export JSON
    _progress(7, "Exporting metadata JSON...")
    t = time.time()

    output = {
        "job_id":            job_id,
        "audio_path":        audio_path,
        "duration_sec":      round(duration_sec, 2),
        "tempo_bpm":         round(float(tempo), 1),
        "onset_count":       len(onsets),
        "stems":             stems,
        "spatial_wav":       spatial_wav,
        "detection_summary": det_summary,
        "motion_summary":    motion_summary,
        "motion_events": [
            {
                "start_sec":           ev["start_sec"],
                "end_sec":             ev["end_sec"],
                "duration_sec":        ev["duration_sec"],
                "type":                ev["type"],
                "spatial_motion":      ev["spatial_motion"],
                "dominant_instrument": ev["dominant_instrument"],
                "confidence":          ev["confidence"],
                "pitch_sequence":      ev["pitch_sequence"][:8],
            }
            for ev in motion_events
        ],
        "spatial_timeline": [
            {
                "time_sec": entry["time_sec"],
                "instruments": [
                    {
                        "label":     i["label"],
                        "azimuth":   i["azimuth"],
                        "distance":  i["distance"],
                        "in_motion": i["in_motion"],
                    }
                    for i in entry["instruments"][:5]
                ]
            }
            for entry in spatial_result["timeline"][::5]
        ],
        "processing_time": round(time.time() - pipeline_start, 1),
    }

    json_path = os.path.join(JSON_DIR, job_id + "_output.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    output["json_path"] = json_path
    _progress(7, "JSON saved: " + json_path
              + " [" + str(round(time.time()-t, 1)) + "s]")

    total_time = round(time.time() - pipeline_start, 1)
    print("[Pipeline] Complete in " + str(total_time) + "s")
    print("[Pipeline] Spatial WAV: " + str(spatial_wav))
    print("[Pipeline] JSON:        " + json_path)

    return output


def get_pipeline_status():
    """Return which pipeline modules are available and working."""
    status = {}
    modules = [
        ("preprocess", "core.preprocess", "load_audio"),
        ("separator",  "core.separator",  "separate_stems"),
        ("detector",   "core.detector",   "detect_instruments"),
        ("motion",     "core.motion",     "detect_motion_events"),
        ("spatial",    "core.spatial",    "assign_spatial_positions"),
        ("hrtf",       "core.hrtf",       "render_binaural"),
    ]
    for name, module, func in modules:
        try:
            mod = __import__(module, fromlist=[func])
            getattr(mod, func)
            status[name] = "ok"
        except Exception as e:
            status[name] = "error: " + str(e)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    status["melband_model"] = "ok" if os.path.exists(
        os.path.join(models_dir, "MelBandRoformer.ckpt")) else "missing"
    status["beats_model"]   = "ok" if os.path.exists(
        os.path.join(models_dir, "BEATs_iter3_plus_AS2M_finetuned_cpt2.pt")) else "missing"
    status["sofa_file"]     = "ok" if os.path.exists(
        os.path.join(models_dir, "mit_kemar_normal_pinna.sofa")) else "missing"

    return status
