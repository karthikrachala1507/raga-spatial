# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - prepare_datasets.py
# Turns the downloaded datasets (download_datasets.py) into Tier 2 training
# clips: 10 s, 16 kHz mono WAVs, one folder per category, in
#   <project>\data\processed\stems_t2\<category>\<dataset>__<source>__<n>.wav
#
# train_separator.py picks new clips up at the start of its next epoch, so this
# can run while training is running.
#
# Usage:
#   python data_tools\prepare_datasets.py              (every dataset that finished downloading)
#   python data_tools\prepare_datasets.py slakh2100    (only the named ones)
#
# Safe to stop and rerun: sources already processed are skipped.

import os
import re
import sys
import csv
import json
import time
import numpy as np
import soundfile as sf
import soxr

DATASETS_DIR = paths.DATASETS_DIR
OUT_DIR      = paths.STEMS_T2_DIR
DONE_FILE    = os.path.join(OUT_DIR, "_processed_sources.json")

TARGET_SR = 16000
CLIP_SEC  = 10.0
CLIP_LEN  = int(TARGET_SR * CLIP_SEC)

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]

# How many 10 s clips to keep per source file, and how loud a window must be
# relative to the loud parts of the same file. Saraga is recorded live, so its
# stems contain bleed from the other players: only its loud windows are kept.
SETTINGS = {
    "saraga_carnatic": {"max_clips": 24, "rel_level": 0.50},
    "musdb18hq":       {"max_clips": 12, "rel_level": 0.25},
    "urmp":            {"max_clips": 12, "rel_level": 0.25},
    "guitarset":       {"max_clips": 3,  "rel_level": 0.25},
    "mridangam_tani":  {"max_clips": 60, "rel_level": 0.25},
    "slakh2100":       {"max_clips": 4,  "rel_level": 0.25},
    "medley_solos":    {"max_clips": 12, "rel_level": 0.0},
}


def log(msg):
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)


def safe_id(text):
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-")[:80]


# ── Window selection ──────────────────────────────────────────────────────────

def window_stats(mono):
    """RMS of a window and the fraction of 50 ms frames that are 'active'."""
    rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
    frame = max(1, len(mono) // 200)
    n = len(mono) // frame
    if n < 10:
        return rms, 0.0
    fr = np.sqrt(np.mean(mono[:n * frame].reshape(n, frame) ** 2, axis=1))
    ref = np.percentile(fr, 95)
    active = float(np.mean(fr > 0.1 * ref)) if ref > 1e-5 else 0.0
    return rms, active


def choose_windows(stats, max_clips, rel_level):
    if not stats:
        return []
    rms = np.array([s[0] for s in stats])
    ref = np.percentile(rms, 90)
    floor = max(rel_level * ref, 1e-3)
    ok = [i for i, (r, a) in enumerate(stats) if r >= floor and a >= 0.35]
    if len(ok) <= max_clips:
        return ok
    picks = np.linspace(0, len(ok) - 1, max_clips).round().astype(int)
    return [ok[i] for i in sorted(set(picks.tolist()))]


def write_clip(mono, sr, category, name):
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR)
    mono = mono.astype(np.float32)
    if len(mono) < CLIP_LEN:
        mono = np.pad(mono, (0, CLIP_LEN - len(mono)))
    mono = mono[:CLIP_LEN]
    peak = float(np.abs(mono).max())
    if peak < 1e-4:
        return False
    mono = mono / peak * 0.9
    out_dir = os.path.join(OUT_DIR, category)
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, name)
    tmp = final + ".tmp"
    sf.write(tmp, mono, TARGET_SR, format="WAV", subtype="PCM_16")
    os.replace(tmp, final)          # training never sees a half-written file
    return True


def process_file(path, dataset, source, category):
    """Cut one stem recording into clips. Returns the number of clips written."""
    cfg = SETTINGS[dataset]
    try:
        with sf.SoundFile(path) as f:
            sr = f.samplerate
            win = int(sr * CLIP_SEC)
            stats = []
            for block in f.blocks(blocksize=win, dtype="float32", always_2d=True):
                if len(block) < win // 3:
                    break                      # ignore a short tail
                stats.append(window_stats(block.mean(axis=1)))
            chosen = choose_windows(stats, cfg["max_clips"], cfg["rel_level"])
            written = 0
            for n, i in enumerate(chosen):
                f.seek(i * win)
                block = f.read(win, dtype="float32", always_2d=True)
                name = "%s__%s__%04d.wav" % (dataset, source, n)
                if write_clip(block.mean(axis=1), sr, category, name):
                    written += 1
            return written
    except Exception as e:
        log("    skipped %s (%s)" % (os.path.basename(path), type(e).__name__))
        return 0


# ── Per-dataset listings: yield (path, source_id, category) ───────────────────

def walk(root, suffixes):
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(suffixes):
                yield os.path.join(dirpath, name)


def list_saraga(root):
    kinds = {
        "vocal": "vocals", "violin": "bowed_strings",
        "mridangam-left": "percussion", "ghatam": "percussion",
    }   # mridangam-right is the same drum on a second mic; vocal-s is a backing voice
    for path in walk(root, (".mp3",)):
        base = os.path.basename(path)[:-4]
        if ".multitrack-" not in base:
            continue
        track, kind = base.rsplit(".multitrack-", 1)
        if kind in kinds:
            yield path, safe_id(track)[:60] + "-" + kind, kinds[kind]


def list_musdb(root):
    kinds = {"vocals.wav": "vocals", "bass.wav": "bass", "drums.wav": "percussion"}
    for path in walk(root, (".wav",)):
        base = os.path.basename(path).lower()
        if base in kinds:
            song = os.path.basename(os.path.dirname(path))
            yield path, safe_id(song)[:60] + "-" + base[:-4], kinds[base]


URMP = {
    "vn": "bowed_strings", "va": "bowed_strings", "vc": "bowed_strings", "db": "bowed_strings",
    "fl": "wind", "ob": "wind", "cl": "wind", "sax": "wind", "bn": "wind",
    "tpt": "wind", "hn": "wind", "tbn": "wind", "tba": "wind",
}


def list_urmp(root):
    for path in walk(root, (".wav",)):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")              # AuSep_1_vn_01_Jupiter
        if len(parts) >= 3 and parts[0] == "AuSep" and parts[2] in URMP:
            yield path, safe_id(base), URMP[parts[2]]


def list_guitarset(root):
    for path in walk(root, (".wav",)):
        yield path, safe_id(os.path.basename(path)[:-4]), "plucked_strings"


def list_tani(root):
    for path in walk(root, (".wav",)):
        yield path, safe_id(os.path.basename(path)[:-4]), "percussion"


def slakh_category(program, is_drum):
    """General MIDI program number -> our category (None = skip)."""
    if is_drum:
        return "percussion"
    p = int(program)
    if p <= 23:   return "keys_synth"        # pianos, mallets, organs, accordion
    if p <= 31:   return "plucked_strings"   # guitars
    if p <= 39:   return "bass"
    if p <= 44:   return "bowed_strings"     # violin, viola, cello, contrabass, tremolo
    if p <= 46:   return "plucked_strings"   # pizzicato, harp
    if p == 47:   return "percussion"        # timpani
    if p <= 51:   return "bowed_strings"     # string ensembles
    if p <= 55:   return None                # synthetic choir, orchestra hit
    if p <= 79:   return "wind"              # brass, reeds, pipes
    if p <= 95:   return "keys_synth"        # synth leads and pads
    if p <= 103:  return None                # sound effects
    if p <= 107:  return "plucked_strings"   # sitar, banjo, shamisen, koto
    if p == 108:  return "keys_synth"        # kalimba
    if p == 109:  return "wind"              # bagpipe
    if p == 110:  return "bowed_strings"     # fiddle
    if p == 111:  return "wind"              # shehnai
    if p <= 119:  return "percussion"        # steel drums, woodblock, taiko, toms
    return None


def list_slakh(root):
    import yaml
    for meta_path in walk(root, ("metadata.yaml",)):
        track_dir = os.path.dirname(meta_path)
        track = os.path.basename(track_dir)
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
        except Exception:
            continue
        for stem_id, info in (meta.get("stems") or {}).items():
            if not info.get("audio_rendered", True):
                continue
            cat = slakh_category(info.get("program_num", 0), info.get("is_drum", False))
            path = os.path.join(track_dir, "stems", stem_id + ".flac")
            if cat and os.path.exists(path):
                yield path, track + "-" + stem_id, cat


MEDLEY = {
    "clarinet": "wind", "flute": "wind", "tenor saxophone": "wind", "trumpet": "wind",
    "violin": "bowed_strings", "piano": "keys_synth",
    "distorted electric guitar": "plucked_strings", "female singer": "vocals",
}


def process_medley(root, done):
    """Medley-solos-DB is 3 s clips: join clips of the same song into 10 s clips."""
    csv_path = os.path.join(root, "Medley-solos-DB_metadata.csv")
    if not os.path.exists(csv_path):
        log("medley_solos: metadata csv missing")
        return 0
    files = {os.path.basename(p): p for p in walk(root, (".wav",))}
    groups = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cat = MEDLEY.get(row["instrument"].strip().lower())
            name = "Medley-solos-DB_%s-%s_%s.wav" % (row["subset"], row["instrument_id"], row["uuid4"])
            if cat and name in files:
                key = (cat, "i%s-song%s" % (row["instrument_id"], row["song_id"]))
                groups.setdefault(key, []).append(files[name])
    total = 0
    for (cat, source), paths in sorted(groups.items()):
        tag = "medley_solos/" + source
        if tag in done:
            continue
        n = 0
        for start in range(0, len(paths) - 3, 4):
            if n >= SETTINGS["medley_solos"]["max_clips"]:
                break
            try:
                parts = []
                for p in paths[start:start + 4]:
                    x, sr = sf.read(p, dtype="float32", always_2d=True)
                    parts.append(x.mean(axis=1))
                if write_clip(np.concatenate(parts), sr, cat,
                              "medley_solos__%s__%04d.wav" % (source, n)):
                    n += 1
            except Exception:
                continue
        done[tag] = n
        total += n
    return total


LISTERS = {
    "saraga_carnatic": list_saraga,
    "musdb18hq":       list_musdb,
    "urmp":            list_urmp,
    "guitarset":       list_guitarset,
    "mridangam_tani":  list_tani,
    "slakh2100":       list_slakh,
}


def save_done(done):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = DONE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(done, f)
    os.replace(tmp, DONE_FILE)


def main():
    wanted = sys.argv[1:] or list(SETTINGS.keys())
    done = {}
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            done = json.load(f)

    for dataset in wanted:
        root = os.path.join(DATASETS_DIR, dataset)
        if not os.path.exists(os.path.join(root, ".done")):
            log(dataset + ": not downloaded yet, skipping")
            continue
        log(dataset + ": preparing clips")
        t0 = time.time()
        if dataset == "medley_solos":
            total = process_medley(root, done)
            save_done(done)
            log("%s: %d new clips" % (dataset, total))
            continue

        per_cat = {}
        n_sources = 0
        last_save = time.time()
        for path, source, category in LISTERS[dataset](root):
            tag = dataset + "/" + source
            if tag in done:
                continue
            n = process_file(path, dataset, source, category)
            done[tag] = n
            n_sources += 1
            per_cat[category] = per_cat.get(category, 0) + n
            if time.time() - last_save > 120:
                save_done(done)
                log("    %d sources so far, clips per category: %s" % (n_sources, per_cat))
                last_save = time.time()
        save_done(done)
        log("%s: %d new sources in %.0f min, clips per category: %s" % (
            dataset, n_sources, (time.time() - t0) / 60, per_cat))

    counts = {}
    for cat in CATEGORIES:
        d = os.path.join(OUT_DIR, cat)
        counts[cat] = len([n for n in os.listdir(d) if n.endswith(".wav")]) if os.path.isdir(d) else 0
    log("Tier 2 clips now on disk: " + str(counts))


if __name__ == "__main__":
    main()
