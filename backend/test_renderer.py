# Raga Spatial - test_renderer.py
# Phase 9: Test full pipeline end-to-end

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from core.renderer import run_full_pipeline, get_pipeline_status

TEST_FILE = "test.flac"
JOB_ID    = "test_render"


def print_sep(n=55):
    print("=" * n)


def progress(step, msg):
    print("  [" + str(step) + "/7] " + msg)


def test_renderer():
    print_sep()
    print("Phase 9: Full Pipeline End-to-End Test")
    print("Input: " + TEST_FILE)
    print_sep()

    # Step 1: Check all modules are available
    print("\n[0] Pipeline status check...")
    status = get_pipeline_status()
    all_ok = True
    for module, state in status.items():
        icon = "OK" if state == "ok" else "FAIL"
        print("  [" + icon + "] " + module.ljust(20) + " " + state)
        if state != "ok":
            all_ok = False

    if not all_ok:
        print("\nSome modules have issues. Fix before continuing.")
        return

    print("\n  All modules OK. Starting pipeline...\n")

    # Step 2: Run full pipeline
    print_sep()
    print("RUNNING FULL PIPELINE")
    print_sep()

    t0 = time.time()
    try:
        result = run_full_pipeline(
            audio_path        = TEST_FILE,
            job_id            = JOB_ID,
            progress_callback = progress,
        )
    except Exception as e:
        import traceback
        print("\nPipeline failed:")
        traceback.print_exc()
        return

    total_time = round(time.time() - t0, 1)

    # Step 3: Print results
    print()
    print_sep()
    print("PIPELINE RESULTS")
    print_sep()

    print("\nAudio info:")
    print("  Duration:     " + str(result["duration_sec"]) + "s")
    print("  Tempo:        " + str(result["tempo_bpm"]) + " BPM")
    print("  Onsets:       " + str(result["onset_count"]))

    print("\nDetected instruments (top 5):")
    for det in result["detection_summary"][:5]:
        bar  = "#" * int(det["avg_confidence"] * 20)
        print("  " + bar + " " + str(round(det["avg_confidence"]*100,1))
              + "% [" + str(det["window_count"]) + "w] " + det["label"])

    print("\nMotion events (" + str(len(result["motion_events"])) + "):")
    for ev in result["motion_events"]:
        print("  " + str(ev["start_sec"]) + "s-" + str(ev["end_sec"]) + "s"
              + " | " + ev["type"]
              + " | " + ev["dominant_instrument"]
              + " | " + ev["spatial_motion"]
              + " | conf=" + str(round(ev["confidence"]*100,0)) + "%")

    print("\nMotion summary:")
    for k, v in result["motion_summary"].items():
        print("  " + str(k).ljust(25) + ": " + str(v))

    print("\nOutput files:")
    print("  Spatial WAV: " + str(result.get("spatial_wav", "none")))
    print("  JSON:        " + str(result.get("json_path", "none")))
    print("  Vocals:      " + str(result["stems"].get("vocals", "none")))
    print("  Instrumental:" + str(result["stems"].get("instrumental", "none")))

    print("\nTotal processing time: " + str(total_time) + "s")
    print("  (for " + str(result["duration_sec"]) + "s of audio)")

    # Check output wav exists and is valid
    wav_path = result.get("spatial_wav")
    if wav_path and os.path.exists(wav_path):
        import soundfile as sf
        data, sr = sf.read(wav_path)
        print()
        print_sep()
        print("SPATIAL WAV VERIFICATION")
        print_sep()
        print("  Path:     " + wav_path)
        print("  Duration: " + str(round(len(data)/sr, 2)) + "s")
        print("  Channels: " + str(data.shape[1] if data.ndim > 1 else 1))
        print("  SR:       " + str(sr) + " Hz")
        import numpy as np
        print("  L peak:   " + str(round(float(abs(data[:,0]).max()), 4)))
        print("  R peak:   " + str(round(float(abs(data[:,1]).max()), 4)))
        l_rms = float(np.sqrt(np.mean(data[:,0]**2)))
        r_rms = float(np.sqrt(np.mean(data[:,1]**2)))
        diff  = round(abs(l_rms - r_rms) / max(l_rms, r_rms) * 100, 1)
        print("  L/R diff: " + str(diff) + "% (>0% means spatial separation working)")
        print()
        print("  --> LISTEN WITH HEADPHONES: " + wav_path)
    else:
        print("\n  WARNING: spatial wav not found at " + str(wav_path))

    print()
    print_sep()
    print("Phase 9 complete!")
    print_sep()


if __name__ == "__main__":
    test_renderer()
