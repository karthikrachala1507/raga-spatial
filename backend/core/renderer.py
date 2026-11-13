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

    # 8-stem path: used whenever the custom separator model is available
    from core import separator_8stem
    if separator_8stem.is_available() and os.environ.get("RAGA_FORCE_2STEM") != "1":
        return _run_8stem_pipeline(
            audio_path, job_id, audio, sr, duration_sec,
            tempo, beats, onsets, pipeline_start, _progress)

    # Step 2: Source separation
    _progress(2, "Separating stems (MelBandRoformer)...")
    t = time.time()
    from core.separator_2stem import separate_stems
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


def _run_8stem_pipeline(audio_path, job_id, audio, sr, duration_sec,
                        tempo, beats, onsets, pipeline_start, _progress):
    """
    8-stem pipeline: separate into 8 category stems, name the instruments in
    each stem, detect motion on the melodic stems, then place every stem at its
    own azimuth and depth layer and render to binaural.
    """
    import soxr
    from core import separator_8stem, spatial_8stem
    from core.spatial_8stem import STEM_LAYOUT, STEM_ORDER

    # Step 2: separation
    _progress(2, "Separating into 8 stems...")
    t = time.time()
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    want_hybrid = (os.environ.get("RAGA_SEP8_MODE", "hybrid") != "pure"
                   and os.path.exists(os.path.join(models_dir, "MelBandRoformer.ckpt")))
    two_stem = {}
    if want_hybrid:
        try:
            _progress(2, "Extracting vocals (MelBandRoformer)...")
            from core.separator_2stem import separate_stems
            two_stem = separate_stems(audio, sr, job_id)
        except Exception as e:
            print("[Pipeline] MelBandRoformer failed (" + str(e) + "), using the 8-stem model alone")
            two_stem = {}
    sep = separator_8stem.separate_8stems(
        audio, sr, job_id,
        vocals_path       = two_stem.get("vocals"),
        instrumental_path = two_stem.get("instrumental"),
        progress          = lambda msg: _progress(2, msg),
    )
    stems  = sep["audio"]
    levels = spatial_8stem.stem_levels(stems, sr)
    active = [s for s in STEM_ORDER if levels[s]["active"]]
    _progress(2, "Separation done (" + sep["mode"] + "): " + str(len(active))
              + " active stems [" + str(round(time.time()-t, 1)) + "s]")

    # Step 3: name the instruments inside each stem
    _progress(3, "Detecting instruments per stem (BEATs)...")
    t = time.time()
    stem_instruments = {s: [] for s in STEM_ORDER}
    if os.environ.get("RAGA_SKIP_DETECT") != "1":
        try:
            from core.detector import detect_instruments
            for name in active:
                a16  = soxr.resample(stems[name], sr, 16000).astype(np.float32)
                peak = float(np.abs(a16).max())
                if peak < 1e-4:
                    continue
                _progress(3, "Detecting instruments in " + STEM_LAYOUT[name]["label"] + "...")
                windows = detect_instruments(a16 / peak * 0.9, sr=16000, stem_name=name)
                stem_instruments[name] = spatial_8stem.name_instruments(name, windows)
        except Exception as e:
            print("[Pipeline] Per-stem detection failed: " + str(e))
    _progress(3, "Detection done [" + str(round(time.time()-t, 1)) + "s]")

    # Step 4: motion on the melodic stems
    _progress(4, "Detecting motion events...")
    t = time.time()
    from core.preprocess import extract_chroma, detect_onsets
    from core.motion import detect_motion_events, summarise_motion
    events_by_stem = {}
    motion_events  = []
    for name in spatial_8stem.MOTION_CARRIERS:
        if not levels[name]["active"] or levels[name]["share"] < 0.03:
            continue
        try:
            events = detect_motion_events(
                chroma     = extract_chroma(stems[name], sr),
                onsets     = detect_onsets(stems[name], sr),
                beats      = beats,
                detections = [],
                sr         = sr,
                hop_length = 512,
            )
        except Exception as e:
            print("[Pipeline] Motion detection failed for " + name + ": " + str(e))
            events = []
        label = (stem_instruments[name][0]["label"] if stem_instruments[name]
                 else STEM_LAYOUT[name]["label"])
        for ev in events:
            ev["dominant_instrument"] = label
            ev["stem"] = name
        events_by_stem[name] = events
        motion_events.extend(events)
    motion_events.sort(key=lambda ev: ev["start_sec"])
    motion_summary = summarise_motion(motion_events, duration_sec)
    # two stems can move at once: count the time when anything moves, once
    covered, last_end = 0.0, 0.0
    for ev in motion_events:
        start = max(ev["start_sec"], last_end)
        if ev["end_sec"] > start:
            covered += ev["end_sec"] - start
            last_end = ev["end_sec"]
    if motion_events:
        motion_summary["motion_coverage_pct"] = round(
            min(covered / max(duration_sec, 1e-6), 1.0) * 100, 1)
    _progress(4, "Motion done: " + str(len(motion_events)) + " events ["
              + str(round(time.time()-t, 1)) + "s]")

    # Step 5: positions and trajectories
    _progress(5, "Assigning spatial positions and depth layers...")
    trajectories, moving = {}, {}
    for name, events in events_by_stem.items():
        if events:
            trajectories[name], moving[name] = spatial_8stem.build_trajectory(
                name, events, duration_sec)

    # Step 6: binaural rendering
    _progress(6, "Rendering binaural audio (HRTF + depth)...")
    t = time.time()
    spatial_wav, binaural_paths = spatial_8stem.render_stems(
        stems, sr, job_id, trajectories, levels,
        progress=lambda msg: _progress(6, msg))
    _progress(6, "HRTF render done [" + str(round(time.time()-t, 1)) + "s]")

    # Step 7: export
    _progress(7, "Exporting metadata JSON...")
    env_step = int(round((1.0 / spatial_8stem.ENVELOPE_HZ) / spatial_8stem.TRAJECTORY_RES_SEC))
    stems8 = []
    for name in STEM_ORDER:
        layout, lv = STEM_LAYOUT[name], levels[name]
        stems8.append({
            "name":         name,
            "label":        layout["label"],
            "layer":        layout["layer"],
            "azimuth":      layout["azimuth"],
            "distance":     layout["distance"],
            "color":        layout["color"],
            "contains":     layout["contains"],
            "active":       lv["active"] and name in binaural_paths,
            "share_pct":    round(lv["share"] * 100, 1),
            "active_pct":   lv["active_pct"],
            "envelope":     lv["envelope"],
            "instruments":  stem_instruments[name],
            "motion_count": len(events_by_stem.get(name, [])),
            "trajectory":   ([round(float(a), 1) for a in trajectories[name][::env_step]]
                             if name in trajectories else None),
        })

    detection_summary = []
    for s in stems8:
        for inst in s["instruments"]:
            detection_summary.append({
                "label":          inst["label"],
                "stem":           s["label"],
                "avg_confidence": inst["avg_confidence"],
                "window_count":   inst["window_count"],
            })
    detection_summary.sort(key=lambda d: d["avg_confidence"], reverse=True)

    # one entry every 0.5 s, same shape the 2-stem UI already understands
    spatial_timeline = []
    for i in range(0, int(duration_sec * spatial_8stem.ENVELOPE_HZ), 2):
        instruments = []
        for s in stems8:
            if not s["active"] or i >= len(s["envelope"]) or s["envelope"][i] < 0.03:
                continue
            az = s["trajectory"][i] if s["trajectory"] and i < len(s["trajectory"]) else s["azimuth"]
            instruments.append({
                "label":     s["label"],
                "azimuth":   az,
                "distance":  s["distance"],
                "in_motion": bool(s["trajectory"]) and abs(az - s["azimuth"]) > 2.5,
            })
        spatial_timeline.append({"time_sec": round(i / spatial_8stem.ENVELOPE_HZ, 2),
                                 "instruments": instruments})

    output = {
        "job_id":            job_id,
        "mode":              "8-stem",
        "separation_mode":   sep["mode"],
        "model_epoch":       sep["model_epoch"],
        "audio_path":        audio_path,
        "duration_sec":      round(duration_sec, 2),
        "tempo_bpm":         round(float(tempo), 1),
        "onset_count":       len(onsets),
        "stems":             sep["paths"],
        "stems8":            stems8,
        "envelope_hz":       spatial_8stem.ENVELOPE_HZ,
        "spatial_wav":       spatial_wav,
        "detection_summary": detection_summary,
        "motion_summary":    motion_summary,
        "motion_events": [
            {
                "start_sec":           ev["start_sec"],
                "end_sec":             ev["end_sec"],
                "duration_sec":        ev["duration_sec"],
                "type":                ev["type"],
                "spatial_motion":      ev["spatial_motion"],
                "dominant_instrument": ev["dominant_instrument"],
                "stem":                ev["stem"],
                "confidence":          ev["confidence"],
                "pitch_sequence":      ev["pitch_sequence"][:8],
            }
            for ev in motion_events
        ],
        "spatial_timeline":  spatial_timeline,
        "processing_time":   round(time.time() - pipeline_start, 1),
    }

    json_path = os.path.join(JSON_DIR, job_id + "_output.json")
    with open(json_path, "w") as f:
        json.dump(output, f)
    output["json_path"] = json_path

    print("[Pipeline] 8-stem pipeline complete in " + str(output["processing_time"]) + "s")
    print("[Pipeline] Spatial WAV: " + str(spatial_wav))
    print("[Pipeline] JSON:        " + json_path)
    return output


def get_pipeline_status():
    """Return which pipeline modules are available and working."""
    status = {}
    modules = [
        ("preprocess", "core.preprocess", "load_audio"),
        ("separator",  "core.separator_2stem",  "separate_stems"),
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


def get_separator_mode():
    """Which separation the next upload will use."""
    from core import separator_8stem
    if separator_8stem.is_available() and os.environ.get("RAGA_FORCE_2STEM") != "1":
        return {"mode": "8-stem", "model": separator_8stem.find_model()}
    return {"mode": "2-stem", "model": None}
