# Raga Spatial - test_hrtf.py
# Phase 8: Test HRTF binaural rendering

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from core.hrtf import (load_sofa, get_hrir_for_azimuth,
                        spatialise_stem, render_binaural_simple)

STEMS_DIR        = os.path.join("outputs", "stems")
JOB_ID           = "test_job"
DETECTIONS_JSON  = os.path.join("outputs", "json", "test_detections_stems.json")
SPATIAL_JSON     = os.path.join("outputs", "json", "test_spatial.json")
OUTPUT_DIR       = os.path.join("outputs", "spatial")


def print_sep(n=55):
    print("=" * n)


def test_hrtf():
    print_sep()
    print("Phase 8: HRTF Binaural Rendering")
    print_sep()

    # Step 1: Load and inspect SOFA file
    print("\n[1] Loading SOFA HRTF file...")
    t0        = time.time()
    sofa_data = load_sofa()
    print("    Load time: " + str(round(time.time()-t0, 2)) + "s")
    print("    Measurements: " + str(len(sofa_data["azimuths"])))
    print("    HRIR length:  " + str(sofa_data["n_taps"]) + " taps")
    print("    Sample rate:  " + str(sofa_data["sr"]) + " Hz")

    # Show azimuth coverage
    az_unique = sorted(set(round(float(a), 0) for a in sofa_data["azimuths"]))
    print("    Azimuth range: " + str(az_unique[0]) + "deg to "
          + str(az_unique[-1]) + "deg ("
          + str(len(az_unique)) + " positions)")

    # Step 2: Test HRIR lookup for key positions
    print("\n[2] Testing HRIR lookup for key positions...")
    test_positions = [
        (0,   "front-center"),
        (45,  "front-right"),
        (90,  "right"),
        (135, "rear-right"),
        (180, "rear-center"),
        (225, "rear-left"),
        (270, "left"),
        (315, "front-left"),
    ]
    for az, desc in test_positions:
        hl, hr = get_hrir_for_azimuth(sofa_data, az)
        itd    = _estimate_itd(hl, hr, sofa_data["sr"])
        print("    az=" + str(az).rjust(3) + "deg (" + desc.ljust(12) + ")"
              + "  HRIR L/R max: "
              + str(round(float(abs(hl).max()), 4)).ljust(7)
              + " / " + str(round(float(abs(hr).max()), 4)).ljust(7)
              + "  ITD: " + str(itd) + "us")

    # Step 3: Test single stem spatialisation
    print("\n[3] Testing single stem spatialisation...")
    import numpy as np
    import soundfile as sf

    vocals_path = os.path.join(STEMS_DIR, JOB_ID + "_vocals.wav")
    if os.path.exists(vocals_path):
        audio, sr = sf.read(vocals_path)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        # Test at front-left (guitar position)
        t0      = time.time()
        binaural = spatialise_stem(audio, azimuth_deg=315.0,
                                   distance=0.35, sofa_data=sofa_data,
                                   stem_sr=sr)
        elapsed = round(time.time()-t0, 2)
        print("    Input:  mono " + str(len(audio)) + " samples (" 
              + str(round(len(audio)/sr, 1)) + "s)")
        print("    Output: stereo " + str(binaural.shape))
        print("    Time:   " + str(elapsed) + "s")
        print("    L peak: " + str(round(float(abs(binaural[0]).max()), 4)))
        print("    R peak: " + str(round(float(abs(binaural[1]).max()), 4)))

        # Confirm left ear stronger for front-left position
        l_rms = float(np.sqrt(np.mean(binaural[0]**2)))
        r_rms = float(np.sqrt(np.mean(binaural[1]**2)))
        correct = "correct" if l_rms > r_rms else "check azimuth convention"
        print("    L RMS > R RMS for front-left: " + correct
              + " (L=" + str(round(l_rms,4)) + " R=" + str(round(r_rms,4)) + ")")
    else:
        print("    Vocals stem not found: " + vocals_path)

    # Step 4: Full simple rendering
    print("\n[4] Running full binaural render (simple fixed positions)...")
    if not os.path.exists(DETECTIONS_JSON):
        print("    ERROR: " + DETECTIONS_JSON + " not found")
        return

    with open(DETECTIONS_JSON) as f:
        det_data = json.load(f)
    summary = det_data.get("summary", [])
    print("    Using " + str(len(summary)) + " detected instruments")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    output_path = render_binaural_simple(
        stems_dir        = STEMS_DIR,
        job_id           = JOB_ID,
        detection_summary= summary,
        sr               = 44100,
    )
    elapsed = round(time.time()-t0, 1)
    print("    Render time: " + str(elapsed) + "s")

    if output_path and os.path.exists(output_path):
        import soundfile as sf
        data, sr = sf.read(output_path)
        print()
        print_sep()
        print("OUTPUT FILE")
        print_sep()
        print("  Path:     " + output_path)
        print("  Shape:    " + str(data.shape) + " (samples x channels)")
        print("  Duration: " + str(round(len(data)/sr, 2)) + "s")
        print("  SR:       " + str(sr) + " Hz")
        print("  L peak:   " + str(round(float(abs(data[:,0]).max()), 4)))
        print("  R peak:   " + str(round(float(abs(data[:,1]).max()), 4)))
        print()
        print("  --> Copy this file to your phone and listen with headphones!")
        print("  --> Guitar should feel like it comes from the front-left")

    print()
    print_sep()
    print("Phase 8 test complete!")
    print_sep()


def _estimate_itd(hrir_l, hrir_r, sr):
    """Estimate interaural time difference in microseconds."""
    import numpy as np
    # Cross-correlate left and right HRIRs
    corr    = np.correlate(hrir_l, hrir_r, mode="full")
    lag     = np.argmax(np.abs(corr)) - (len(hrir_r) - 1)
    itd_us  = round(lag / sr * 1e6, 1)
    return itd_us


if __name__ == "__main__":
    test_hrtf()
