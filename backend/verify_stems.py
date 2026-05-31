# Raga Spatial - verify_stems.py
# Verifies, cleans, and standardizes all downloaded stem clips
# Converts to 16kHz mono WAV, removes silence-only clips, trims to 10s

import os
import sys
import json
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

DATA_DIR      = r"D:\raga-spatial-data\raw_stems"
PROCESSED_DIR = r"D:\raga-spatial-data\processed\stems"

# Standardization settings
TARGET_SR     = 16000   # 16kHz for BEATs compatibility
TARGET_DUR    = 10.0    # seconds per clip
MIN_RMS       = 0.005   # minimum RMS energy (filter silence)
MAX_RMS       = 0.95    # maximum RMS (filter clipping)
OVERLAP       = 5.0     # seconds overlap when splitting long clips

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]


def process_audio_file(input_path, output_dir, category, file_idx):
    """
    Load audio, standardize, split into 10s clips, save.
    Returns number of valid clips extracted.
    """
    try:
        audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        return 0, "load_error: " + str(e)

    if len(audio) < TARGET_SR * 2:
        return 0, "too_short"

    # Check overall quality
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < MIN_RMS:
        return 0, "silence"
    if rms > MAX_RMS:
        return 0, "clipping"

    # Split into TARGET_DUR clips with OVERLAP
    clip_samples = int(TARGET_SR * TARGET_DUR)
    hop_samples  = int(TARGET_SR * (TARGET_DUR - OVERLAP))
    clips_saved  = 0

    pos = 0
    clip_num = 0
    while pos + clip_samples <= len(audio):
        clip = audio[pos:pos + clip_samples]

        # Check clip quality
        clip_rms = float(np.sqrt(np.mean(clip ** 2)))
        if clip_rms < MIN_RMS:
            pos += hop_samples
            continue

        # Normalize clip
        clip = clip / (np.max(np.abs(clip)) + 1e-8) * 0.9

        # Save
        filename    = category + "_" + str(file_idx).zfill(5) + "_" + str(clip_num).zfill(3) + ".wav"
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, clip, TARGET_SR)

        clips_saved += 1
        clip_num    += 1
        pos         += hop_samples

    return clips_saved, "ok"


def verify_category(category):
    """Process all files in a category."""
    input_dir  = os.path.join(DATA_DIR, category)
    output_dir = os.path.join(PROCESSED_DIR, category)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print("[" + category + "] Input dir not found: " + input_dir)
        return 0

    # Get all audio files
    audio_files = []
    for f in os.listdir(input_dir):
        if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            audio_files.append(os.path.join(input_dir, f))

    if not audio_files:
        print("[" + category + "] No audio files found")
        return 0

    print("[" + category + "] Processing " + str(len(audio_files)) + " files...")

    total_clips = 0
    errors      = {}
    existing    = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])

    if existing > 0:
        print("[" + category + "] Already processed: " + str(existing) + " clips")

    for i, filepath in enumerate(tqdm(audio_files, desc=category)):
        clips, status = process_audio_file(filepath, output_dir, category, i)
        total_clips  += clips
        if status != "ok":
            errors[status] = errors.get(status, 0) + 1

    print("[" + category + "] Clips extracted: " + str(total_clips))
    if errors:
        print("[" + category + "] Rejected: " + str(errors))

    return total_clips


def main():
    print("=" * 60)
    print("Raga Spatial — Stem Verification and Standardization")
    print("Converting to 16kHz mono WAV, 10s clips")
    print("Output: " + PROCESSED_DIR)
    print("=" * 60)

    summary = {}
    total   = 0

    for category in CATEGORIES:
        count          = verify_category(category)
        summary[category] = count
        total         += count

        # Save progress
        with open(r"D:\raga-spatial-data\verify_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    for category, count in summary.items():
        print("  " + category.ljust(20) + ": " + str(count) + " clips")
    print("  TOTAL".ljust(22) + ": " + str(total) + " clips")

    # Estimate storage
    size_mb = total * TARGET_SR * 2 * TARGET_DUR / (1024 * 1024)
    print("\nEstimated processed storage: " + str(round(size_mb / 1024, 1)) + " GB")


if __name__ == "__main__":
    main()
