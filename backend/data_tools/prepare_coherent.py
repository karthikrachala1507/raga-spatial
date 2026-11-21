# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # for prepare_datasets
import paths

# Raga Spatial - prepare_coherent.py
# Cuts the MULTITRACK datasets into time-aligned stems, so training can build
# "coherent" mixes: instruments that really played together, in the same key
# and tempo, at the same moment.
#
# prepare_datasets.py cuts every stem on its own and normalises each clip, so
# its clips can only be combined at random. That gives correct but unmusical
# mixes. Real songs are harder: the parts move together. This tool keeps
# what makes them hard:
#   - every stem of a track is cut at the SAME 10 s windows
#   - all stems of a window share ONE gain, so their true balance is kept
#
# Output: <project>\data\processed\stems_aligned\<category>\
#           <dataset>__<track>__<source>__w<window>.wav     (16 kHz mono)
#
# Usage (from backend\):
#   python data_tools\prepare_coherent.py              every multitrack dataset
#   python data_tools\prepare_coherent.py urmp musdb18hq
#
# Safe to stop and rerun: tracks already done are skipped.

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import soxr

import prepare_datasets as pd

OUT_DIR   = paths.STEMS_ALIGNED_DIR
DONE_FILE = os.path.join(OUT_DIR, "_processed_tracks.json")

TARGET_SR = 16000
CLIP_SEC  = 10.0

# windows kept per track, and how many tracks to use (None = all)
SETTINGS = {
    "urmp":            {"max_windows": 8,  "max_tracks": None},
    "musdb18hq":       {"max_windows": 12, "max_tracks": None},
    "saraga_carnatic": {"max_windows": 16, "max_tracks": None},
    "slakh2100":       {"max_windows": 6,  "max_tracks": 900},
    "saraga_audiovisual": {"max_windows": 16, "max_tracks": None},
}
MIN_REL_DB = -40.0     # a stem this far below the window's mix counts as absent


def tracks_urmp(root):
    groups = {}
    for path in pd.walk(root, (".wav",)):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        if len(parts) >= 3 and parts[0] == "AuSep" and parts[2] in pd.URMP:
            piece = pd.safe_id(os.path.basename(os.path.dirname(path)))
            groups.setdefault(piece, []).append((path, pd.safe_id(base), pd.URMP[parts[2]]))
    for track in sorted(groups):
        yield track, groups[track]


def tracks_musdb(root):
    kinds = {"vocals.wav": "vocals", "bass.wav": "bass", "drums.wav": "percussion"}
    groups = {}
    for path in pd.walk(root, (".wav",)):
        base = os.path.basename(path).lower()
        if base in kinds:
            song = pd.safe_id(os.path.basename(os.path.dirname(path)))[:60]
            groups.setdefault(song, []).append((path, song + "-" + base[:-4], kinds[base]))
    for track in sorted(groups):
        yield track, groups[track]


def tracks_saraga(root):
    kinds = {"vocal": "vocals", "violin": "bowed_strings",
             "mridangam-left": "percussion", "ghatam": "percussion"}
    groups = {}
    for path in pd.walk(root, (".mp3",)):
        base = os.path.basename(path)[:-4]
        if ".multitrack-" not in base:
            continue
        name, kind = base.rsplit(".multitrack-", 1)
        if kind in kinds:
            track = pd.safe_id(name)[:60]
            groups.setdefault(track, []).append((path, track + "-" + kind, kinds[kind]))
    for track in sorted(groups):
        yield track, groups[track]


def tracks_saraga_av(root):
    kinds = {"vocal": "vocals", "violin": "bowed_strings",
             "mridangam-left": "percussion", "ghatam": "percussion"}
    groups = {}
    for path in pd.walk(root, (".wav",)):
        base = os.path.basename(path)[:-4]
        if ".multitrack-" not in base:
            continue
        _, kind = base.rsplit(".multitrack-", 1)
        if kind in kinds:
            track = pd.safe_id(os.path.relpath(os.path.dirname(path), root))[:70]
            groups.setdefault(track, []).append((path, track + "-" + kind, kinds[kind]))
    for track in sorted(groups):
        yield track, groups[track]


def tracks_slakh(root):
    import yaml
    for meta_path in pd.walk(root, ("metadata.yaml",)):
        track_dir = os.path.dirname(meta_path)
        track = os.path.basename(track_dir)
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
        except Exception:
            continue
        stems = []
        for stem_id, info in (meta.get("stems") or {}).items():
            if not info.get("audio_rendered", True):
                continue
            cat = pd.slakh_category(info.get("program_num", 0), info.get("is_drum", False))
            path = os.path.join(track_dir, "stems", stem_id + ".flac")
            if cat and os.path.exists(path):
                stems.append((path, track + "-" + stem_id, cat))
        yield track, stems


LISTERS = {
    "urmp": tracks_urmp,
    "musdb18hq": tracks_musdb,
    "saraga_carnatic": tracks_saraga,
    "slakh2100": tracks_slakh,
    "saraga_audiovisual": tracks_saraga_av,
}


def process_track(dataset, track, stems, max_windows):
    """Cut one track. Returns the number of aligned windows written."""
    files = []
    try:
        for path, source, category in stems:
            files.append(sf.SoundFile(path))
        sr = files[0].samplerate
        if any(f.samplerate != sr for f in files):
            return 0
        win = int(sr * CLIP_SEC)

        # pass 1: how loud is each stem in each window
        rms = []
        for f in files:
            r = []
            for block in f.blocks(blocksize=win, dtype="float32", always_2d=True):
                if len(block) < win:
                    break
                r.append(float(np.sqrt(np.mean(block.mean(axis=1) ** 2))))
            rms.append(np.array(r))
        n = min(len(r) for r in rms)
        if n == 0:
            return 0
        active = np.zeros((len(files), n), dtype=bool)
        for s, r in enumerate(rms):
            if len(r) and r.max() > 0:
                active[s] = r[:n] >= max(0.25 * np.percentile(r, 90), 1e-3)

        cats = [c for _, _, c in stems]
        def n_cats(i):
            return len({cats[s] for s in range(len(files)) if active[s, i]})
        good = [i for i in range(n) if n_cats(i) >= 2]            # at least two different stems
        if not good:
            return 0
        if len(good) > max_windows:
            picks = np.linspace(0, len(good) - 1, max_windows).round().astype(int)
            good = [good[i] for i in sorted(set(picks.tolist()))]

        # pass 2: cut those windows from every stem that is playing in them
        written = 0
        for i in good:
            clips = {}
            for s, f in enumerate(files):
                if not active[s, i]:
                    continue
                f.seek(i * win)
                x = f.read(win, dtype="float32", always_2d=True).mean(axis=1)
                if sr != TARGET_SR:
                    x = soxr.resample(x, sr, TARGET_SR)
                clips[s] = x.astype(np.float32)
            if len(clips) < 2:
                continue
            length = min(len(x) for x in clips.values())
            mix = np.sum([x[:length] for x in clips.values()], axis=0)
            peak = float(np.abs(mix).max())
            mix_rms = float(np.sqrt(np.mean(mix ** 2)))
            if peak < 1e-3:
                continue
            scale = 0.9 / peak                                   # ONE gain for the whole window
            floor = mix_rms * 10 ** (MIN_REL_DB / 20)
            for s, x in clips.items():
                if float(np.sqrt(np.mean(x ** 2))) < floor:
                    continue
                _, source, category = stems[s]
                out_dir = os.path.join(OUT_DIR, category)
                os.makedirs(out_dir, exist_ok=True)
                name = "%s__%s__%s__w%04d.wav" % (dataset, track, source, i)
                tmp = os.path.join(out_dir, name + ".tmp")
                sf.write(tmp, x[:length] * scale, TARGET_SR, format="WAV", subtype="PCM_16")
                os.replace(tmp, os.path.join(out_dir, name))
            written += 1
        return written
    except Exception as e:
        pd.log("    skipped %s/%s (%s)" % (dataset, track, type(e).__name__))
        return 0
    finally:
        for f in files:
            f.close()


def save_done(done):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DONE_FILE + ".tmp", "w") as f:
        json.dump(done, f)
    os.replace(DONE_FILE + ".tmp", DONE_FILE)


def main():
    wanted = sys.argv[1:] or list(SETTINGS.keys())
    done = {}
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            done = json.load(f)

    for dataset in wanted:
        if dataset not in SETTINGS:
            pd.log(dataset + ": not a multitrack dataset, skipping")
            continue
        root = os.path.join(paths.DATASETS_DIR, dataset)
        if not os.path.exists(os.path.join(root, ".done")):
            pd.log(dataset + ": not downloaded yet, skipping")
            continue
        cfg = SETTINGS[dataset]
        pd.log(dataset + ": cutting time-aligned windows")
        t0 = time.time()
        n_tracks = n_windows = 0
        last = time.time()
        already = sum(1 for k in done if k.startswith(dataset + "/"))
        for track, stems in LISTERS[dataset](root):
            if cfg["max_tracks"] is not None and already + n_tracks >= cfg["max_tracks"]:
                break
            tag = dataset + "/" + track
            if tag in done or len(stems) < 2:
                continue
            w = process_track(dataset, track, stems, cfg["max_windows"])
            done[tag] = w
            n_tracks += 1
            n_windows += w
            if time.time() - last > 120:
                save_done(done)
                pd.log("    %d tracks, %d aligned windows so far" % (n_tracks, n_windows))
                last = time.time()
        save_done(done)
        pd.log("%s: %d new tracks, %d aligned windows in %.0f min" % (
            dataset, n_tracks, n_windows, (time.time() - t0) / 60))

    total = {}
    for cat in pd.CATEGORIES:
        d = os.path.join(OUT_DIR, cat)
        total[cat] = len([n for n in os.listdir(d) if n.endswith(".wav")]) if os.path.isdir(d) else 0
    pd.log("aligned stems now on disk: " + str(total))


if __name__ == "__main__":
    main()
