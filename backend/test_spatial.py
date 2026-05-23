# Raga Spatial - test_spatial.py
# Phase 7: Test spatial position assignment

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from core.preprocess   import load_audio, normalize_audio, get_tempo_and_beats
from core.motion       import build_motion_timeline
from core.spatial      import (assign_spatial_positions, get_position,
                                azimuth_to_description, INSTRUMENT_POSITIONS)

STEMS_JSON    = os.path.join("outputs", "json", "test_detections_stems.json")
MOTION_JSON   = os.path.join("outputs", "json", "test_motion.json")
OUTPUT_JSON   = os.path.join("outputs", "json", "test_spatial.json")


def print_sep(char="=", n=55):
    print(char * n)


def test_spatial():
    print_sep()
    print("Phase 7: Spatial Position Assignment")
    print_sep()

    # Load detections from Phase 5
    print("\n[1] Loading Phase 5 detections...")
    if not os.path.exists(STEMS_JSON):
        print("ERROR: " + STEMS_JSON + " not found. Run test_detector.py first.")
        return
    with open(STEMS_JSON) as f:
        det_data = json.load(f)
    detections = det_data.get("merged", [])
    print("    Loaded " + str(len(detections)) + " detection windows")

    # Load motion events from Phase 6
    print("\n[2] Loading Phase 6 motion events...")
    if not os.path.exists(MOTION_JSON):
        print("ERROR: " + MOTION_JSON + " not found. Run test_motion.py first.")
        return
    with open(MOTION_JSON) as f:
        motion_data = json.load(f)
    motion_events   = motion_data.get("motion_events", [])
    total_duration  = motion_data.get("duration_sec", 30.0)
    print("    Loaded " + str(len(motion_events)) + " motion events")
    print("    Duration: " + str(total_duration) + "s")

    # Run spatial assignment
    print("\n[3] Running spatial assignment...")
    t0     = time.time()
    result = assign_spatial_positions(
        detections         = detections,
        motion_events      = motion_events,
        total_duration_sec = total_duration,
        resolution_sec     = 0.5,   # 0.5s resolution for test (use 0.1 for production)
    )
    elapsed = round(time.time() - t0, 2)
    print("    Time: " + str(elapsed) + "s")

    # Print static position map
    print()
    print_sep()
    print("STATIC INSTRUMENT POSITION MAP")
    print_sep()
    print("  (positions when not in motion)")
    print()
    print("  {:<28} {:>8}  {:>8}  {}".format(
        "Instrument", "Azimuth", "Distance", "Direction"))
    print("  " + "-"*60)

    detected_labels = set()
    for dw in detections:
        for det in dw["detections"]:
            detected_labels.add(det["label"])

    for label in sorted(detected_labels):
        from core.spatial import INSTRUMENT_POSITIONS, DEFAULT_POSITION
        pos  = INSTRUMENT_POSITIONS.get(label, None)
        if pos is None:
            continue
        desc = azimuth_to_description(pos["azimuth"])
        print("  {:<28} {:>7}deg  {:>7}    {}".format(
            pos.get("label", label),
            pos["azimuth"],
            pos["distance"],
            desc
        ))

    # Print timeline sample every 2 seconds
    print()
    print_sep()
    print("SPATIAL TIMELINE (sampled every 2s)")
    print_sep()

    timeline = result["timeline"]
    prev_sec = -2.0

    for entry in timeline:
        t = entry["time_sec"]
        if t - prev_sec < 1.99:
            continue
        prev_sec = t

        instruments = entry["instruments"]
        if not instruments:
            print("\n  " + str(t) + "s — (no detected instruments)")
            continue

        print("\n  " + str(t) + "s:")
        for instr in instruments[:5]:
            motion_tag = " [MOVING]" if instr["in_motion"] else ""
            desc       = azimuth_to_description(instr["azimuth"])
            print("    {:<22} az={:>6}deg  dist={}  {}{}".format(
                instr["label"],
                instr["azimuth"],
                instr["distance"],
                desc,
                motion_tag
            ))

    # Print motion trajectory detail
    if motion_events:
        print()
        print_sep()
        print("MOTION TRAJECTORIES")
        print_sep()
        for i, ev in enumerate(motion_events):
            print("\n  Event " + str(i+1) + ": " + ev["dominant_instrument"]
                  + " (" + str(ev["start_sec"]) + "s - " + str(ev["end_sec"]) + "s)")
            print("  Spatial motion: " + ev["spatial_motion"])
            print("  Trajectory:")

            # Sample trajectory at 25%, 50%, 75% progress
            from core.spatial import _motion_azimuth, INSTRUMENT_POSITIONS, DEFAULT_POSITION
            base_pos = INSTRUMENT_POSITIONS.get(
                ev["dominant_instrument"], DEFAULT_POSITION
            )
            base_az  = base_pos["azimuth"]
            for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
                az   = _motion_azimuth(base_az, ev["spatial_motion"], pct)
                desc = azimuth_to_description(az)
                t_at = round(ev["start_sec"] + pct * ev["duration_sec"], 1)
                print("    t=" + str(t_at) + "s  progress=" + str(int(pct*100))
                      + "%  az=" + str(round(az, 0)) + "deg  (" + desc + ")")

    # Summary
    print()
    print_sep()
    print("SUMMARY")
    print_sep()
    n_instruments = len(set(
        instr["label"]
        for entry in timeline
        for instr in entry["instruments"]
    ))
    n_motion_steps = sum(
        1 for entry in timeline
        if any(i["in_motion"] for i in entry["instruments"])
    )
    print("  Timeline steps:       " + str(len(timeline)))
    print("  Unique instruments:   " + str(n_instruments))
    print("  Steps with motion:    " + str(n_motion_steps)
          + " (" + str(round(n_motion_steps/max(len(timeline),1)*100,1)) + "%)")
    print("  Motion events:        " + str(len(motion_events)))

    # Save JSON (trim timeline to every 0.5s for file size)
    os.makedirs(os.path.join("outputs", "json"), exist_ok=True)
    # Keep full timeline but limit instruments list per step to top 6
    slim_timeline = []
    for entry in timeline:
        slim_timeline.append({
            "time_sec":    entry["time_sec"],
            "instruments": entry["instruments"][:6],
        })
    result["timeline"] = slim_timeline
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print("\n  Saved: " + OUTPUT_JSON)

    print()
    print_sep()
    print("Phase 7 test complete!")
    print_sep()


if __name__ == "__main__":
    test_spatial()
