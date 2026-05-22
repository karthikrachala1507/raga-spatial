# Raga Spatial - test_detector.py
# Phase 5 updated: tests both raw mix AND stem-aware detection

import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.preprocess import load_audio, resample_for_beats
from core.detector import detect_instruments, detect_from_stems, summarise_detections

TEST_FILE        = "test.flac"
VOCALS_STEM      = os.path.join("outputs", "stems", "test_job_vocals.wav")
INSTR_STEM       = os.path.join("outputs", "stems", "test_job_instrumental.wav")
OUT_MIX_JSON     = os.path.join("outputs", "json", "test_detections_mix.json")
OUT_STEMS_JSON   = os.path.join("outputs", "json", "test_detections_stems.json")


def print_separator():
    print("=" * 55)


def print_window_results(results, label=""):
    if label:
        print_separator()
        print(label)
        print_separator()
    for window in results:
        stem_tag = " [" + window.get("stem","") + "]" if window.get("stem") else ""
        print("\n  " + str(window["start_sec"]) + "s-" + str(window["end_sec"]) + "s" + stem_tag + ":")
        if window["detections"]:
            for det in window["detections"][:6]:
                bar = "#" * int(det["confidence"] * 30)
                src = " <" + det.get("stem","") + ">" if det.get("stem") else ""
                print("    " + bar + " " + str(round(det["confidence"]*100,1)) + "% -- " + det["label"] + src)
        else:
            print("    (nothing above threshold)")


def print_summary(results, label=""):
    summary = summarise_detections(results, top_n=10)
    if label:
        print_separator()
        print(label)
        print_separator()
    for item in summary:
        bar = "#" * int(item["avg_confidence"] * 30)
        print("  " + bar + " " + str(round(item["avg_confidence"]*100,1)) + "% avg"
              + " [" + str(item["window_count"]) + " windows] -- " + item["label"])


def test_detector():
    os.makedirs(os.path.join("outputs", "json"), exist_ok=True)

    # ─── MODE 1: Raw mix detection ───────────────────────────────────────────
    print_separator()
    print("MODE 1: Detection on raw mixed audio")
    print_separator()

    audio, sr  = load_audio(TEST_FILE)
    audio_16k  = resample_for_beats(audio, sr)
    print("16kHz shape: " + str(audio_16k.shape))

    t0          = time.time()
    mix_results = detect_instruments(audio_16k, sr=16000, stem_name="mix")
    print("\nTime (mix): " + str(round(time.time()-t0, 1)) + "s")

    print_window_results(mix_results, "PER-WINDOW (mix)")
    print_summary(mix_results, "SUMMARY (mix)")

    with open(OUT_MIX_JSON, "w") as f:
        json.dump({"mode": "mix", "windows": mix_results,
                   "summary": summarise_detections(mix_results)}, f, indent=2)
    print("\nSaved: " + OUT_MIX_JSON)

    # ─── MODE 2: Stem-aware detection ────────────────────────────────────────
    if os.path.exists(VOCALS_STEM) and os.path.exists(INSTR_STEM):
        print()
        print_separator()
        print("MODE 2: Stem-aware detection (vocals + instrumental separately)")
        print_separator()

        t0           = time.time()
        stem_results = detect_from_stems(VOCALS_STEM, INSTR_STEM)
        print("\nTime (stems): " + str(round(time.time()-t0, 1)) + "s")

        print_window_results(stem_results["merged"], "PER-WINDOW MERGED (stems)")
        print_summary(stem_results, "SUMMARY MERGED (stems)")

        print()
        print_separator()
        print("COMPARISON: mix vs stems")
        print_separator()
        mix_summary   = summarise_detections(mix_results,   top_n=5)
        stems_summary = summarise_detections(stem_results,  top_n=5)
        print("\nTop 5 from raw mix:")
        for s in mix_summary:
            print("  " + s["label"] + " (" + str(round(s["avg_confidence"]*100,1)) + "%)")
        print("\nTop 5 from stems:")
        for s in stems_summary:
            print("  " + s["label"] + " (" + str(round(s["avg_confidence"]*100,1)) + "%)")

        with open(OUT_STEMS_JSON, "w") as f:
            json.dump({"mode": "stems",
                       "vocals":       stem_results["vocals"],
                       "instrumental": stem_results["instrumental"],
                       "merged":       stem_results["merged"],
                       "summary":      summarise_detections(stem_results)}, f, indent=2)
        print("\nSaved: " + OUT_STEMS_JSON)
    else:
        print("\n[!] Stem files not found -- skipping stem-aware detection")
        print("    Expected: " + VOCALS_STEM)
        print("    Expected: " + INSTR_STEM)
        print("    Run test_separator.py first to generate stems")

    print()
    print_separator()
    print("Detection test complete!")
    print_separator()


if __name__ == "__main__":
    test_detector()
