# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - corpus_benchmark.py
# A larger yardstick than the 11 reference songs: a random sample of the owner's own song
# folder (default 60 songs, 60 s from the middle of each). The truth is BEATs on the original
# mix (independent of the separator). Each model is scored with the same rules as the
# reference scorecard, plus how often it lets one stem swallow the song.
# The songs are only read and cut locally; nothing is copied into the repo or uploaded.
# Output: data\evaluation\corpus\CORPUS_BENCHMARK.md  (+ corpus_stats.json: what the songs contain)
# Usage (from backend\):  python training\corpus_benchmark.py [N] [folder]
#         models scored: separator_v5_best, v6, v7, v8 (those that exist)
import os, sys, json, random, subprocess
import numpy as np
import soxr

from core import separator_8stem as sep
from training.analyze_reference_songs import ffmpeg_exe
from training import scorecard_reference_songs as sc

FOLDER = r"C:\Users\racha\Desktop\music\antha manolle bro"
OUT = os.path.join(paths.DATA_DIR, "evaluation", "corpus")
EXC = 60
TAGS = ["v5", "v6", "v7", "v8"]


def pick(folder, n):
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".mp3", ".m4a")))
    random.Random(11).shuffle(files)
    return files[:n]


def excerpt(path):
    """60 s from the middle as 16 kHz mono floats (None if the file cannot be read)."""
    try:
        dur = subprocess.run([ffmpeg_exe(), "-i", path], capture_output=True).stderr.decode("utf-8", "ignore")
        import re
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", dur)
        total = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 120
        start = max(0.0, total / 2 - EXC / 2)
        raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-ss", str(start), "-t", str(EXC), "-i", path,
                              "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        a = np.frombuffer(raw, dtype=np.float32)
        return a if len(a) > 16000 * 20 else None
    except Exception as e:
        print("skip", os.path.basename(path), e)
        return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    folder = sys.argv[2] if len(sys.argv) > 2 else FOLDER
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "excerpt_tags.json")
    tags = json.load(open(cache)) if os.path.exists(cache) else {}
    from core.detector import detect_instruments

    clips = {}
    for path in pick(folder, n):
        a = excerpt(path)
        if a is None:
            continue
        key = os.path.basename(path)
        clips[key] = a
        if key in tags:
            continue
        windows = detect_instruments(a, 16000, "mix")
        hits = {c: 0 for c in sc.CATS}
        for w in windows:
            seen = {sc.category_of(d["label"]) for d in w["detections"]} - {None}
            for c in seen:
                hits[c] += 1
        nw = max(len(windows), 1)
        tags[key] = {c: round(100 * hits[c] / nw, 1) for c in sc.CATS}
        json.dump(tags, open(cache, "w"), indent=1)
        print("tagged", key, flush=True)

    # what the songs contain (the "data process" answer for this folder)
    stats = {c: round(float(np.mean([t[c] for t in tags.values()])), 1) for c in sc.CATS}
    common = {c: int(sum(t[c] >= 30 for t in tags.values())) for c in sc.CATS}
    json.dump({"songs": len(tags), "mean_presence_pct": stats, "songs_with_it_at_30pct": common},
              open(os.path.join(OUT, "corpus_stats.json"), "w"), indent=1)

    lines = ["# Corpus benchmark", "",
             "%d songs from the owner's folder, %d s from the middle of each. Truth = BEATs on the original mix." % (len(clips), EXC), "",
             "What the songs contain (songs where the category is audible in at least 30%% of windows): %s" % common, "",
             "| model | mean recall | flags per song | songs with a DOMINANT stem | MISSED | PHANTOM | CATCH-ALL | mean top-stem share |",
             "|---|---|---|---|---|---|---|---|"]
    for tag in TAGS:
        ck = os.path.join(paths.CHECKPOINTS_DIR, "separator_%s_best.pt" % tag)
        if not os.path.exists(ck):
            continue
        os.environ["RAGA_SEP8_MODEL"] = ck
        sep._MODEL_CACHE.clear()
        dev = sep.pick_device()
        model, _ = sep.load_model(dev)
        rec, fl, dom, miss, ph, ca, top = [], [], 0, 0, 0, 0, []
        for key, a in clips.items():
            est = sep.infer_16k(model, a, dev)
            mono44 = soxr.resample(a, 16000, 44100).astype(np.float32)
            _, shares = sep.fullband_stems(mono44, est, 44100)
            mj = {"stems": [{"stem": c, "share_pct": float(shares[i]) * 100} for i, c in enumerate(model.categories)]}
            _, flags, recall = sc.judge(mj, tags[key], [])
            rec.append(recall); fl.append(len(flags))
            dom += any(f.startswith("DOMINANT") for f in flags)
            miss += sum(f.startswith("MISSED") for f in flags)
            ph += sum(f.startswith("PHANTOM") for f in flags)
            ca += any(f.startswith("CATCH") for f in flags)
            top.append(max(s["share_pct"] for s in mj["stems"]))
        lines.append("| %s | %.1f%% | %.2f | %d | %d | %d | %d | %.0f%% |" % (
            tag, np.mean(rec), np.mean(fl), dom, miss, ph, ca, np.mean(top)))
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    open(os.path.join(OUT, "CORPUS_BENCHMARK.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
