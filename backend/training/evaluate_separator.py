# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# Raga Spatial - evaluate_separator.py
# Listening test: split one real song into the 8 stems with the trained model
# and write them as WAV files you can open and listen to.
#
# Usage:
#   python training\evaluate_separator.py                         (test5.mp3)
#   python training\evaluate_separator.py --song "D:\path\to\song.mp3"
#   python training\evaluate_separator.py --model separator_v3_latest.pt
#
# Output: <project>\data\evaluation\<song name>\<stem>.wav (44.1 kHz)
# This is the custom model alone ("pure" mode). The app's default "hybrid"
# mode takes the vocals from MelBandRoformer instead.

import os
import sys
import shutil
import argparse
import numpy as np

import paths

EVAL_DIR     = paths.EVAL_DIR
CHECKPOINTS_DIR   = paths.CHECKPOINTS_DIR
DEFAULT_SONG = os.path.join(paths.REAL_SONGS_DIR, "test5.mp3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default=DEFAULT_SONG)
    parser.add_argument("--model", default=None,
                        help="checkpoint file name in " + CHECKPOINTS_DIR + " (default: best)")
    args = parser.parse_args()

    if args.model:
        path = args.model if os.path.isabs(args.model) else os.path.join(CHECKPOINTS_DIR, args.model)
        os.environ["RAGA_SEP8_MODEL"] = path

    from core import separator_8stem
    from core.preprocess import load_audio, normalize_audio

    if not separator_8stem.is_available():
        print("No trained model found yet. Train first: start_training.bat (project folder)")
        return
    if not os.path.exists(args.song):
        print("Song not found: " + args.song)
        return

    name    = os.path.splitext(os.path.basename(args.song))[0]
    out_dir = os.path.join(EVAL_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading " + args.song)
    audio, sr = load_audio(args.song)
    audio     = normalize_audio(audio)
    result    = separator_8stem.separate_8stems(audio, sr, "eval_" + name, progress=print)

    total = sum(float(np.sum(a.astype(np.float64) ** 2)) for a in result["audio"].values())
    print("\nModel epoch " + str(result["model_epoch"]) + " | stems written to " + out_dir)
    for stem, src in result["paths"].items():
        energy = float(np.sum(result["audio"][stem].astype(np.float64) ** 2))
        shutil.move(src, os.path.join(out_dir, stem + ".wav"))
        print("  %-16s %5.1f%% of the song's energy" % (stem, 100 * energy / max(total, 1e-12)))
    print("\nListen to each file: a good stem contains its own instruments and little else.")


if __name__ == "__main__":
    main()
