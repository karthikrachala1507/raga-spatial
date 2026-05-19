"""
Raga Spatial — Test preprocess.py
Run this to verify preprocessing works on a real audio file.

Usage:
    python test_preprocess.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocess import preprocess_audio, get_tempo_and_beats
import numpy as np


def test_preprocess():
    # ── Find a test audio file ────────────────────────────────────────
    # Put any MP3 or WAV file in backend folder and update this path
    test_files = [
        "test.mp3",
        "test.wav",
        "outputs/uploads/"  # Use any uploaded file
    ]

    test_file = None
    for f in test_files:
        if os.path.isfile(f):
            test_file = f
            break
        elif os.path.isdir(f):
            files = os.listdir(f)
            if files:
                test_file = os.path.join(f, files[0])
                break

    if not test_file:
        print("ERROR: No test audio file found!")
        print("Put any MP3 or WAV file in the backend folder named 'test.mp3'")
        return

    print(f"Testing with: {test_file}")
    print("=" * 50)

    # ── Run preprocessing ─────────────────────────────────────────────
    result = preprocess_audio(test_file)

    # ── Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("PREPROCESSING RESULTS:")
    print("=" * 50)
    print(f"Duration:           {result['duration']:.2f} seconds")
    print(f"Sample rate:        {result['sr']} Hz")
    print(f"Audio shape:        {result['audio'].shape}")
    print(f"Audio 16kHz shape:  {result['audio_16k'].shape}")
    print(f"Mel spec shape:     {result['mel_spectrogram'].shape}")
    print(f"Chroma shape:       {result['chroma'].shape}")
    print(f"Onset count:        {len(result['onset_times'])}")
    print(f"Short windows:      {len(result['short_windows'])} x {3.0}s")
    print(f"Long windows:       {len(result['long_windows'])} x {12.0}s")

    # ── Tempo ─────────────────────────────────────────────────────────
    tempo, beats = get_tempo_and_beats(result["audio"], result["sr"])
    print(f"Tempo:              {tempo:.1f} BPM")
    print(f"Beat count:         {len(beats)}")

    # ── Sample window info ────────────────────────────────────────────
    if result["short_windows"]:
        w = result["short_windows"][0]
        print(f"\nFirst short window: {w['start']:.2f}s → {w['end']:.2f}s")

    print("\n✓ Preprocessing module working correctly!")
    print("Ready for Phase 4 — BS-RoFormer source separation")


if __name__ == "__main__":
    test_preprocess()
