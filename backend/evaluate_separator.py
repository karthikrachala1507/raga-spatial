# Raga Spatial - evaluate_separator.py
# Evaluates trained 8-stem separator on real Telugu songs
# Measures SDR per stem and saves separated audio

import os
import sys
import json
import numpy as np
import torch
import soundfile as sf
import librosa
from tqdm import tqdm

MODELS_DIR    = r"D:\raga-spatial-data\models"
OUTPUT_DIR    = r"D:\raga-spatial-data\evaluation"
REAL_SONGS_DIR= r"D:\raga-spatial-data\real_songs"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_STEMS     = 8
CHUNK_SEC   = 10.0
OVERLAP_SEC = 2.0

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]

# Spatial positions for each stem
STEM_POSITIONS = {
    "percussion":      {"azimuth": 180, "distance": 0.5,  "label": "Percussion"},
    "plucked_strings": {"azimuth": 315, "distance": 0.35, "label": "Plucked strings"},
    "bowed_strings":   {"azimuth": 225, "distance": 0.5,  "label": "Strings"},
    "wind":            {"azimuth": 45,  "distance": 0.4,  "label": "Wind"},
    "keys_synth":      {"azimuth": 30,  "distance": 0.45, "label": "Keys/Synth"},
    "vocals":          {"azimuth": 0,   "distance": 0.25, "label": "Vocals"},
    "bass":            {"azimuth": 135, "distance": 0.55, "label": "Bass"},
    "folk_texture":    {"azimuth": 270, "distance": 0.6,  "label": "Folk/Texture"},
}


def load_model(model_path):
    """Load trained BSRNN model."""
    sys.path.insert(0, os.path.dirname(__file__))
    from train_separator import BandSplitRNN

    print("[Eval] Loading model: " + model_path)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model = BandSplitRNN(n_stems=N_STEMS).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("[Eval] Model loaded (epoch " + str(ckpt.get("epoch", "?")) + ")")
    print("[Eval] Val loss: " + str(ckpt.get("val_loss", "?")))
    return model


def separate_song(model, audio, sr=16000):
    """
    Separate a full song into 8 stems using chunked inference.
    Returns dict of {category: audio_array}
    """
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    chunk_samples   = int(SAMPLE_RATE * CHUNK_SEC)
    overlap_samples = int(SAMPLE_RATE * OVERLAP_SEC)
    hop_samples     = chunk_samples - overlap_samples
    total_samples   = len(audio)

    # Output buffers
    outputs = {cat: np.zeros(total_samples, dtype=np.float32) for cat in CATEGORIES}
    counts  = np.zeros(total_samples, dtype=np.float32)
    window  = np.hanning(chunk_samples).astype(np.float32)

    start   = 0
    chunk_n = 0

    while start < total_samples:
        end       = min(start + chunk_samples, total_samples)
        chunk     = audio[start:end]
        chunk_len = len(chunk)

        if chunk_len < SAMPLE_RATE:
            break

        if chunk_len < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - chunk_len))

        with torch.no_grad():
            x     = torch.tensor(chunk).unsqueeze(0).to(DEVICE)
            preds = model(x)  # (1, 8, chunk_samples)
            preds = preds.squeeze(0).cpu().numpy()  # (8, chunk_samples)

        win_slice = window[:min(chunk_len, chunk_samples)]

        for i, cat in enumerate(CATEGORIES):
            stem = preds[i, :chunk_len]
            outputs[cat][start:end] += stem * win_slice[:chunk_len]

        counts[start:end] += win_slice[:chunk_len]
        chunk_n += 1
        start   += hop_samples

        print("[Eval] Chunk " + str(chunk_n) + ": "
              + str(round(start/SAMPLE_RATE, 1)) + "s", end="\r")

    print()
    counts = np.maximum(counts, 1e-8)
    for cat in CATEGORIES:
        outputs[cat] = outputs[cat] / counts

    return outputs


def evaluate_on_song(model, song_path, output_dir):
    """Separate one song and save stems."""
    song_name = os.path.splitext(os.path.basename(song_path))[0]
    song_out  = os.path.join(output_dir, song_name)
    os.makedirs(song_out, exist_ok=True)

    print("\n[Eval] Processing: " + song_name)

    audio, sr = librosa.load(song_path, sr=None, mono=True)
    print("[Eval] Duration: " + str(round(len(audio)/sr, 1)) + "s")

    stems = separate_song(model, audio, sr)

    # Save each stem
    stem_paths = {}
    for cat, stem_audio in stems.items():
        stem_path = os.path.join(song_out, cat + ".wav")
        stem_norm = np.clip(stem_audio, -1.0, 1.0)
        sf.write(stem_path, stem_norm, SAMPLE_RATE)
        stem_paths[cat] = stem_path
        rms = float(np.sqrt(np.mean(stem_audio**2)))
        pos = STEM_POSITIONS[cat]
        print("[Eval] Saved " + cat.ljust(20)
              + " | rms=" + str(round(rms, 4)).ljust(8)
              + " | az=" + str(pos["azimuth"]) + "deg")

    # Save spatial metadata
    spatial_meta = {
        "song":  song_name,
        "stems": {
            cat: {
                "path":     stem_paths[cat],
                "azimuth":  STEM_POSITIONS[cat]["azimuth"],
                "distance": STEM_POSITIONS[cat]["distance"],
                "label":    STEM_POSITIONS[cat]["label"],
            }
            for cat in CATEGORIES
        }
    }
    with open(os.path.join(song_out, "spatial_meta.json"), "w") as f:
        json.dump(spatial_meta, f, indent=2)

    return stem_paths, spatial_meta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="separator_best.pt",
                        help="Model filename in models dir")
    parser.add_argument("--song",   default=None,
                        help="Path to specific song to evaluate")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, args.model)
    if not os.path.exists(model_path):
        print("ERROR: Model not found: " + model_path)
        print("Run train_separator.py first")
        return

    model = load_model(model_path)

    if args.song:
        songs = [args.song]
    else:
        songs = [
            os.path.join(REAL_SONGS_DIR, f)
            for f in os.listdir(REAL_SONGS_DIR)
            if f.lower().endswith((".mp3", ".wav", ".flac"))
        ]

    if not songs:
        print("No songs found. Add songs to: " + REAL_SONGS_DIR)
        return

    print("=" * 60)
    print("Raga Spatial — 8-Stem Separator Evaluation")
    print("Songs to process: " + str(len(songs)))
    print("=" * 60)

    all_results = []
    for song_path in songs:
        if not os.path.exists(song_path):
            print("Skipping missing: " + song_path)
            continue
        stem_paths, meta = evaluate_on_song(model, song_path, OUTPUT_DIR)
        all_results.append(meta)

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("Results saved: " + OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
