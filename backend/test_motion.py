# Raga Spatial - test_motion.py
# Phase 6: Test motion event detection

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from core.preprocess import (load_audio, normalize_audio, extract_chroma,
                              detect_onsets, get_tempo_and_beats)
from core.motion import (detect_motion_events, build_motion_timeline,
                          summarise_motion, NOTE_NAMES)

TEST_FILE      = "test.flac"
DETECTIONS_JSON = os.path.join("outputs", "json", "test_detections_stems.json")
OUTPUT_JSON     = os.path.join("outputs", "json", "test_motion.json")


def print_sep(char="=", width=55):
    print(char * width)


def test_motion():
    print_sep()
    print("Phase 6: Motion Event Detection")
    print("Test file: " + TEST_FILE)
    print_sep()

    # Step 1: Preprocess
    print("\n[1] Preprocessing...")
    audio, sr = load_audio(TEST_FILE)
    audio     = normalize_audio(audio)
    chroma    = extract_chroma(audio, sr)
    onsets    = detect_onsets(audio, sr)
    tempo, beats = get_tempo_and_beats(audio, sr)
    duration  = len(audio) / sr

    print("    Chroma: "   + str(chroma.shape))
    print("    Onsets: "   + str(len(onsets)))
    print("    Tempo:  "   + str(round(float(tempo), 1)) + " BPM")
    print("    Duration: " + str(round(duration, 1)) + "s")

    # Step 2: Load detections from Phase 5
    print("\n[2] Loading Phase 5 detections...")
    if os.path.exists(DETECTIONS_JSON):
        with open(DETECTIONS_JSON) as f:
            det_data = json.load(f)
        detections = det_data.get("merged", [])
        print("    Loaded " + str(len(detections)) + " detection windows from stems")
    else:
        print("    WARNING: " + DETECTIONS_JSON + " not found")
        print("    Running without detection context (instrument names will be generic)")
        detections = []

    # Step 3: Run motion detection
    print("\n[3] Running motion detection...")
    t0 = time.time()
    motion_events = detect_motion_events(
        chroma    = chroma,
        onsets    = onsets,
        beats     = beats,
        detections= detections,
        sr        = sr,
        hop_length= 512,
    )
    elapsed = round(time.time() - t0, 2)
    print("\n    Time: " + str(elapsed) + "s")

    # Step 4: Build timeline
    print("\n[4] Building motion timeline...")
    timeline = build_motion_timeline(motion_events, duration)

    # Step 5: Print results
    print()
    print_sep()
    print("MOTION EVENTS DETECTED: " + str(len(motion_events)))
    print_sep()

    if not motion_events:
        print("  No motion events detected above threshold.")
        print("  This may mean:")
        print("  - Song intro is mostly static/held chords")
        print("  - Threshold needs tuning for this song style")
    else:
        for i, ev in enumerate(motion_events):
            print()
            print("  Event " + str(i+1) + ":")
            print("    Time:        " + str(ev["start_sec"]) + "s -> " + str(ev["end_sec"]) + "s"
                  + " (" + str(ev["duration_sec"]) + "s)")
            print("    Type:        " + ev["type"])
            print("    Spatial:     " + ev["spatial_motion"])
            print("    Instrument:  " + ev["dominant_instrument"])
            print("    Confidence:  " + str(round(ev["confidence"]*100, 1)) + "%")
            print("    Onset/sec:   " + str(ev.get("onset_density", "?")))
            pitches = ev["pitch_sequence"][:12]
            print("    Pitches:     " + " -> ".join(pitches)
                  + (" ..." if len(ev["pitch_sequence"]) > 12 else ""))

    # Step 6: Timeline view
    print()
    print_sep()
    print("SECOND-BY-SECOND TIMELINE")
    print_sep()
    print("  Time  | Motion | Direction      | Instrument          | Conf")
    print("  ------|--------|----------------|---------------------|-----")
    for entry in timeline:
        t       = str(entry["time_sec"]).ljust(5)
        motion  = "YES   " if entry["in_motion"] else "no    "
        direc   = (entry["spatial_motion"]      or "-").ljust(16)
        instr   = (entry["dominant_instrument"] or "-").ljust(20)
        conf    = str(round(entry["confidence"]*100,0)) + "%" if entry["in_motion"] else "-"
        print("  " + t + " | " + motion + " | " + direc + " | " + instr + " | " + conf)

    # Step 7: Summary
    print()
    print_sep()
    print("SUMMARY")
    print_sep()
    summary = summarise_motion(motion_events, duration)
    for k, v in summary.items():
        print("  " + str(k).ljust(25) + ": " + str(v))

    # Step 8: Save JSON
    os.makedirs(os.path.join("outputs", "json"), exist_ok=True)
    output = {
        "test_file":     TEST_FILE,
        "duration_sec":  round(duration, 2),
        "motion_events": motion_events,
        "timeline":      timeline,
        "summary":       summary,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: " + OUTPUT_JSON)

    print()
    print_sep()
    print("Phase 6 test complete!")
    print_sep()


if __name__ == "__main__":
    test_motion()
