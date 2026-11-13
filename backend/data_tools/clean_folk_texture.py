# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - clean_folk_texture.py
# The Tier 1 folk_texture clips were cut from whole folk songs (voices, melody
# and drums together), which taught the separator to put complete music into
# that stem. This tool keeps only the clips that are folk PERCUSSION: the
# BEATs detector must hear percussion and must not hear a voice or a melodic
# instrument.
#
# Nothing is deleted. Rejected clips are MOVED to
#   <project>\data\processed\rejected\folk_texture\
# and can be moved back.
#
# Usage (from backend\):
#   python data_tools\clean_folk_texture.py            score every clip, show what would happen
#   python data_tools\clean_folk_texture.py --apply    move the rejected clips out of training
#   python data_tools\clean_folk_texture.py --restore  move every rejected clip back
#
# Scoring is saved in folk_texture_scores.json, so reruns are instant.

import os
import sys
import json
import shutil
import numpy as np

FOLK_DIR     = os.path.join(paths.STEMS_T1_DIR, "folk_texture")
REJECT_DIR   = os.path.join(paths.REJECTED_DIR, "folk_texture")
SCORES_FILE  = os.path.join(paths.DATA_DIR, "processed", "folk_texture_scores.json")

# A clip is kept when percussion is clearly present and neither a voice nor a
# melodic instrument is. Scores are BEATs probabilities (0..1), the highest
# seen in any 4 s window of the clip.
MIN_PERCUSSION = 0.20
MAX_VOICE      = 0.10
MAX_MELODIC    = 0.15

VOICE_WORDS = ("singing", "speech", "vocal", "choir", "chant", "rapping", "humming",
               "yodel", "a capella", "conversation", "narration", "shout", "whoop",
               "children", "crowd", "cheer", "applause", "laugh")
PERC_WORDS  = ("drum", "percussion", "tabla", "tambourine", "cymbal", "snare", "rimshot",
               "timpani", "wood block", "maraca", "cowbell", "gong", "bass drum", "hi-hat")
MELODIC_WORDS = ("flute", "wind instrument", "saxophone", "clarinet", "trumpet", "trombone",
                 "brass", "french horn", "bagpipe", "harmonica", "accordion", "organ",
                 "piano", "keyboard", "synthesizer", "guitar", "plucked", "sitar", "banjo",
                 "mandolin", "harp", "violin", "fiddle", "cello", "bowed", "string section",
                 "orchestra", "double bass", "bass guitar")

WINDOW  = 64000            # 4 s at 16 kHz, what the detector is used with
STARTS  = (0, 32000, 64000, 96000)
BATCH   = 8                # clips per GPU batch


def clip_source(name):
    return os.path.splitext(name)[0].rsplit("_", 1)[0]


def family_indices():
    from core.detector import _get_label_map, BLOCKLIST_INDICES
    label_map = _get_label_map()
    fam = {"voice": [], "percussion": [], "melodic": []}
    for idx, label in label_map.items():
        if idx in BLOCKLIST_INDICES:
            continue
        low = label.lower()
        if any(w in low for w in VOICE_WORDS):
            fam["voice"].append(idx)
        elif any(w in low for w in PERC_WORDS):
            fam["percussion"].append(idx)
        elif any(w in low for w in MELODIC_WORDS):
            fam["melodic"].append(idx)
    return fam, label_map


def score_clips(names, folder_of):
    """BEATs family scores for each clip: {name: [voice, percussion, melodic]}."""
    import torch
    import soundfile as sf
    from core.detector import load_beats_model, DEVICE
    fam, label_map = family_indices()
    print("labels used -> voice: %d, percussion: %d, melodic: %d" % (
        len(fam["voice"]), len(fam["percussion"]), len(fam["melodic"])))
    model = load_beats_model()
    out = {}
    for b in range(0, len(names), BATCH):
        batch = names[b:b + BATCH]
        windows = []
        for name in batch:
            audio, _ = sf.read(os.path.join(folder_of[name], name), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.pad(audio, (0, max(0, STARTS[-1] + WINDOW - len(audio))))
            windows.extend(audio[s:s + WINDOW] for s in STARTS)
        with torch.no_grad():
            probs, _ = model.extract_features(
                torch.from_numpy(np.stack(windows)).to(DEVICE), padding_mask=None)
        probs = probs.float().cpu().numpy().reshape(len(batch), len(STARTS), -1)
        for i, name in enumerate(batch):
            peak = probs[i].max(axis=0)                     # best window per label
            out[name] = [round(float(peak[fam[k]].max()), 4)
                         for k in ("voice", "percussion", "melodic")]
        if (b // BATCH) % 100 == 0:
            print("  scored %d / %d" % (min(b + BATCH, len(names)), len(names)), flush=True)
    return out


def keep(score):
    voice, perc, melodic = score
    return perc >= MIN_PERCUSSION and voice <= MAX_VOICE and melodic <= MAX_MELODIC


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"

    if mode == "--restore":
        moved = 0
        if os.path.isdir(REJECT_DIR):
            for name in os.listdir(REJECT_DIR):
                shutil.move(os.path.join(REJECT_DIR, name), os.path.join(FOLK_DIR, name))
                moved += 1
        print("moved %d clips back into training" % moved)
        return

    # every folk clip, wherever it currently is
    folder_of = {}
    for folder in (FOLK_DIR, REJECT_DIR):
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.endswith(".wav"):
                    folder_of[name] = folder
    names = sorted(folder_of)

    scores = {}
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            scores = json.load(f)
    todo = [n for n in names if n not in scores]
    if todo:
        print("scoring %d clips with BEATs..." % len(todo))
        scores.update(score_clips(todo, folder_of))
        with open(SCORES_FILE + ".tmp", "w") as f:
            json.dump(scores, f)
        os.replace(SCORES_FILE + ".tmp", SCORES_FILE)

    arr = np.array([scores[n] for n in names])
    kept = [n for n in names if keep(scores[n])]
    src_all  = {clip_source(n) for n in names}
    src_kept = {clip_source(n) for n in kept}
    per_src = {}
    for n in kept:
        per_src[clip_source(n)] = per_src.get(clip_source(n), 0) + 1

    print("\nfolk_texture clips: %d from %d recordings" % (len(names), len(src_all)))
    for i, fam in enumerate(("voice", "percussion", "melodic")):
        print("  %-10s score: median %.2f | 25%% %.2f | 75%% %.2f | 90%% %.2f" % (
            fam, np.median(arr[:, i]), np.percentile(arr[:, i], 25),
            np.percentile(arr[:, i], 75), np.percentile(arr[:, i], 90)))
    print("  has a voice (> %.2f): %d%% | has percussion (>= %.2f): %d%% | has melody (> %.2f): %d%%" % (
        MAX_VOICE, 100 * np.mean(arr[:, 0] > MAX_VOICE),
        MIN_PERCUSSION, 100 * np.mean(arr[:, 1] >= MIN_PERCUSSION),
        MAX_MELODIC, 100 * np.mean(arr[:, 2] > MAX_MELODIC)))
    print("\nKEEP (percussion, no voice, no melody): %d clips (%d%%) from %d recordings" % (
        len(kept), round(100 * len(kept) / max(len(names), 1)), len(src_kept)))
    print("REJECT: %d clips; %d recordings lose every clip" % (
        len(names) - len(kept), len(src_all) - len(src_kept)))
    if per_src:
        v = sorted(per_src.values(), reverse=True)
        print("kept clips per recording: largest %s | median %d" % (v[:5], v[len(v) // 2]))

    if mode != "--apply":
        print("\nNothing was moved. Run with --apply to move the rejected clips out of training.")
        return

    os.makedirs(REJECT_DIR, exist_ok=True)
    to_reject = to_keep = 0
    for n in names:
        want = FOLK_DIR if keep(scores[n]) else REJECT_DIR
        if folder_of[n] != want:
            shutil.move(os.path.join(folder_of[n], n), os.path.join(want, n))
            if want == REJECT_DIR:
                to_reject += 1
            else:
                to_keep += 1
    print("\nmoved %d clips to %s" % (to_reject, REJECT_DIR))
    if to_keep:
        print("moved %d clips back into training" % to_keep)
    print("folk_texture now has %d training clips" % len(
        [n for n in os.listdir(FOLK_DIR) if n.endswith(".wav")]))


if __name__ == "__main__":
    main()
