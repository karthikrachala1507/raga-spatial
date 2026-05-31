# Raga Spatial - augment_mix.py
# Generates synthetic mixes from collected stems
# Each mix has 3-6 random stems combined at random volumes
# Output: mix.wav + individual target stems for training

import os
import sys
import json
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

PROCESSED_DIR = r"D:\raga-spatial-data\processed\stems"
MIXES_DIR     = r"D:\raga-spatial-data\synthetic_mixes"

# Mix generation settings
TARGET_SR          = 16000
CLIP_DURATION      = 10.0    # seconds
CLIPS_PER_MIX_MIN  = 2       # minimum stems per mix
CLIPS_PER_MIX_MAX  = 6       # maximum stems per mix
N_MIXES_TIER1      = 5000    # mixes for Tier 1 training
N_MIXES_TIER2      = 40000   # mixes for Tier 2 training
VOLUME_MIN         = 0.3     # minimum stem volume in mix
VOLUME_MAX         = 1.0     # maximum stem volume in mix

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]

# Category index for one-hot encoding
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}


def load_all_stems():
    """Load file paths for all processed stems."""
    stems = {}
    for category in CATEGORIES:
        cat_dir = os.path.join(PROCESSED_DIR, category)
        if not os.path.exists(cat_dir):
            print("[Augment] Warning: " + cat_dir + " not found")
            stems[category] = []
            continue
        files = [os.path.join(cat_dir, f)
                 for f in os.listdir(cat_dir) if f.endswith(".wav")]
        stems[category] = files
        print("[Augment] " + category + ": " + str(len(files)) + " clips")
    return stems


def load_clip(path):
    """Load a single audio clip."""
    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        target_len = int(TARGET_SR * CLIP_DURATION)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        return audio
    except Exception:
        return None


def apply_augmentation(audio):
    """Apply random augmentation to a stem clip."""
    # Random pitch shift (small amount)
    if random.random() < 0.3:
        steps = random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=TARGET_SR, n_steps=steps)

    # Random time stretch (small amount)
    if random.random() < 0.2:
        rate  = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        target_len = int(TARGET_SR * CLIP_DURATION)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

    # Random gain
    gain  = random.uniform(0.7, 1.0)
    audio = audio * gain

    return audio


def generate_mix(stems_dict, mix_idx, output_dir):
    """Generate one synthetic mix with random stems."""
    # Pick random number of stems
    n_stems     = random.randint(CLIPS_PER_MIX_MIN, CLIPS_PER_MIX_MAX)

    # Pick random categories (no repeats)
    available   = [cat for cat, files in stems_dict.items() if len(files) > 0]
    if len(available) < 2:
        return False

    chosen_cats = random.sample(available, min(n_stems, len(available)))

    # Load and augment each stem
    mix         = np.zeros(int(TARGET_SR * CLIP_DURATION), dtype=np.float32)
    stem_data   = {}
    label       = np.zeros(len(CATEGORIES), dtype=np.float32)

    for cat in chosen_cats:
        filepath = random.choice(stems_dict[cat])
        audio    = load_clip(filepath)
        if audio is None:
            continue

        # Apply augmentation
        audio = apply_augmentation(audio)

        # Random volume for this stem in the mix
        volume = random.uniform(VOLUME_MIN, VOLUME_MAX)

        # Add to mix
        mix += audio * volume

        # Store individual stem (normalized)
        stem_norm        = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        stem_data[cat]   = stem_norm
        label[CAT_TO_IDX[cat]] = 1.0

    if not stem_data:
        return False

    # Normalize mix to prevent clipping
    mix_peak = np.max(np.abs(mix))
    if mix_peak > 0.95:
        mix = mix / mix_peak * 0.95

    # Save mix directory
    mix_dir = os.path.join(output_dir, "mix_" + str(mix_idx).zfill(6))
    os.makedirs(mix_dir, exist_ok=True)

    # Save mix
    sf.write(os.path.join(mix_dir, "mix.wav"), mix, TARGET_SR)

    # Save individual stems (zero for absent categories)
    target_len = int(TARGET_SR * CLIP_DURATION)
    for cat in CATEGORIES:
        stem_audio = stem_data.get(cat, np.zeros(target_len, dtype=np.float32))
        sf.write(os.path.join(mix_dir, cat + ".wav"), stem_audio, TARGET_SR)

    # Save label
    with open(os.path.join(mix_dir, "label.json"), "w") as f:
        json.dump({
            "mix_idx":    mix_idx,
            "categories": chosen_cats,
            "label":      label.tolist(),
        }, f)

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=1,
                        help="Training tier: 1=5000 mixes, 2=40000 mixes")
    args = parser.parse_args()

    n_mixes = N_MIXES_TIER1 if args.tier == 1 else N_MIXES_TIER2

    print("=" * 60)
    print("Raga Spatial — Synthetic Mix Generator")
    print("Tier " + str(args.tier) + ": generating " + str(n_mixes) + " mixes")
    print("Output: " + MIXES_DIR)
    print("=" * 60)

    # Load all stems
    print("\n[Augment] Loading stem file paths...")
    stems_dict = load_all_stems()

    available_cats = [c for c, f in stems_dict.items() if len(f) > 0]
    print("[Augment] Available categories: " + str(len(available_cats)) + "/8")

    if len(available_cats) < 2:
        print("ERROR: Need at least 2 categories with clips. Run collect_stems.py first.")
        return

    # Check existing mixes
    os.makedirs(MIXES_DIR, exist_ok=True)
    existing = len([d for d in os.listdir(MIXES_DIR)
                    if os.path.isdir(os.path.join(MIXES_DIR, d))])
    print("[Augment] Existing mixes: " + str(existing))
    start_idx = existing

    if existing >= n_mixes:
        print("[Augment] Already have enough mixes for Tier " + str(args.tier))
        return

    # Generate mixes
    print("[Augment] Generating " + str(n_mixes - existing) + " new mixes...")
    success = 0
    failed  = 0

    for i in tqdm(range(start_idx, n_mixes), desc="Generating mixes"):
        ok = generate_mix(stems_dict, i, MIXES_DIR)
        if ok:
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("MIX GENERATION COMPLETE")
    print("=" * 60)
    print("  Generated: " + str(success))
    print("  Failed:    " + str(failed))
    print("  Total:     " + str(existing + success))

    # Estimate storage
    size_per_mix = TARGET_SR * 2 * CLIP_DURATION * 9  # 9 files per mix (1 mix + 8 stems)
    total_size_gb = (existing + success) * size_per_mix / (1024 ** 3)
    print("\nEstimated storage: " + str(round(total_size_gb, 1)) + " GB")


if __name__ == "__main__":
    main()
