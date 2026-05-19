# Raga Spatial - Test separator.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.preprocess import preprocess_audio
from core.separator import separate_stems
import time

def test_separator():
    test_file = None
    for f in ["test.flac", "test.mp3", "test.wav"]:
        if os.path.exists(f):
            test_file = f
            break

    if not test_file:
        print("ERROR: No test audio file found!")
        return

    print("Testing separation on: " + test_file)
    print("=" * 50)

    print("\n[1] Preprocessing audio...")
    preprocessed = preprocess_audio(test_file)
    audio = preprocessed["audio"]
    sr = preprocessed["sr"]
    print("    Duration: " + str(round(preprocessed["duration"], 1)) + "s")

    print("\n[2] Running MelBandRoformer separation...")
    print("    This will take 1-3 minutes on RTX 3060...")
    start = time.time()

    stems = separate_stems(audio, sr, job_id="test_job")

    elapsed = time.time() - start
    print("\n    Time taken: " + str(round(elapsed, 1)) + " seconds")

    print("\n" + "=" * 50)
    print("SEPARATION RESULTS:")
    print("=" * 50)
    for stem_name, stem_path in stems.items():
        if os.path.exists(stem_path):
            size = os.path.getsize(stem_path) / (1024 * 1024)
            print("  " + stem_name + " -> " + stem_path + " (" + str(round(size, 1)) + " MB)")
        else:
            print("  " + stem_name + " -> MISSING!")

    print("\nSeparation complete!")
    print("Check outputs/stems/ folder for separated audio files")

if __name__ == "__main__":
    test_separator()
