# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - scorecard_reference_songs.py
# Real-song scorecard: compares what the 8-stem model found in each reference
# song with what is actually in the song.
#
# "What is actually in the song" comes from two places, because the internet
# has almost no per-song instrument credits for these tracks:
#   1. BEATs run on the ORIGINAL MIX (independent of the separator), mapped to
#      the 8 categories: how much of the song each category is audible in.
#   2. truth.json (data\evaluation\reference_songs\truth.json): categories the
#      owner or a credit source says are present. Edit it to add more.
# And one physical check: how much of the mix energy is below 250 Hz.
#
# Flags per song and per model:
#   MISSED     a category is audible in the mix (>=30% of windows) but its stem
#              got less than 3% of the energy
#   PHANTOM    a stem got >=15% of the energy but the category is audible in
#              less than 10% of the mix windows
#   CATCH-ALL  folk_texture took more than 30%
#   DOMINANT   one stem took more than 60%
#   LOW-END    >=30% of the mix energy is below 250 Hz but bass+percussion got
#              less than 10%
#
# Usage (from backend\):  python training\scorecard_reference_songs.py
# Needs the per-model JSONs from analyze_reference_songs.py (v3: <name>.json,
# other models: RAGA_REF_TAG=_v4 -> <name>_v4.json).

import os
import sys
import json
import subprocess
import numpy as np

from training.analyze_reference_songs import SONGS, OUT_DIR, find_song, ffmpeg_exe

TAGS_DIR = os.path.join(OUT_DIR, "mix_tags")
TRUTH    = os.path.join(OUT_DIR, "truth.json")
MODELS   = [("v3", ""), ("v4", "_v4"), ("v5", "_v5"), ("v6", "_v6"), ("v7", "_v7"), ("v8", "_v8")]
CATS = ["percussion", "plucked_strings", "bowed_strings", "wind", "keys_synth", "vocals", "bass"]

# keyword (lower case, in the AudioSet label) -> category; first match wins
KEYWORDS = [
    ("bass drum", "percussion"), ("bass guitar", "bass"), ("double bass", "bass"), ("bass", "bass"),
    ("drum", "percussion"), ("percussion", "percussion"), ("tabla", "percussion"), ("mridangam", "percussion"), ("ghatam", "percussion"), ("cymbal", "percussion"),
    ("hi-hat", "percussion"), ("snare", "percussion"), ("tambourine", "percussion"), ("conga", "percussion"),
    ("bongo", "percussion"), ("maraca", "percussion"), ("clapping", "percussion"), ("shaker", "percussion"),
    ("mandolin", "plucked_strings"), ("guitar", "plucked_strings"), ("banjo", "plucked_strings"),
    ("sitar", "plucked_strings"), ("harp", "plucked_strings"), ("ukulele", "plucked_strings"),
    ("pizzicato", "plucked_strings"), ("plucked", "plucked_strings"),
    ("violin", "bowed_strings"), ("fiddle", "bowed_strings"), ("cello", "bowed_strings"),
    ("bowed", "bowed_strings"), ("string section", "bowed_strings"), ("orchestra", "bowed_strings"),
    ("flute", "wind"), ("wind instrument", "wind"), ("saxophone", "wind"), ("clarinet", "wind"),
    ("oboe", "wind"), ("harmonica", "wind"), ("whistl", "wind"), ("trumpet", "wind"), ("trombone", "wind"),
    ("brass", "wind"), ("bagpipe", "wind"), ("shehnai", "wind"), ("recorder", "wind"),
    ("piano", "keys_synth"), ("keyboard", "keys_synth"), ("organ", "keys_synth"),
    ("synthesizer", "keys_synth"), ("harpsichord", "keys_synth"), ("harmonium", "keys_synth"),
    ("accordion", "keys_synth"),
    ("sing", "vocals"), ("vocal", "vocals"), ("choir", "vocals"), ("chant", "vocals"),
    ("humming", "vocals"), ("voice", "vocals"), ("male", "vocals"), ("female", "vocals"),
]


def category_of(label):
    low = label.lower()
    for word, cat in KEYWORDS:
        if word in low:
            return cat
    return None


def mix_tags(part, path):
    """BEATs on the original mix -> {category: % of windows it is audible in}."""
    os.makedirs(TAGS_DIR, exist_ok=True)
    key = "".join(c if c.isalnum() else "_" for c in part)[:30]
    cache = os.path.join(TAGS_DIR, key + ".json")
    if os.path.exists(cache):
        return json.load(open(cache))
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    from core.detector import detect_instruments
    windows = detect_instruments(audio, 16000, "mix")
    hits = {c: 0 for c in CATS}
    top = {}
    for w in windows:
        seen = set()
        for d in w["detections"]:
            cat = category_of(d["label"])
            if cat:
                seen.add(cat)
                top[d["label"]] = top.get(d["label"], 0) + 1
        for c in seen:
            hits[c] += 1
    n = max(len(windows), 1)
    out = {"presence_pct": {c: round(100 * hits[c] / n, 1) for c in CATS},
           "top_labels": sorted(top.items(), key=lambda kv: -kv[1])[:8], "windows": len(windows)}
    json.dump(out, open(cache, "w"), indent=1)
    return out


def judge(model_json, presence, truth_cats):
    shares = {s["stem"]: s["share_pct"] for s in model_json["stems"]}
    # a model with a separate hand-percussion stem is scored on drums + hand drums together
    shares["percussion"] = shares.get("percussion", 0) + shares.get("hand_percussion", 0)
    flags = []
    for c in CATS:
        if presence[c] >= 30 and shares.get(c, 0) < 3:
            flags.append("MISSED " + c)
    # BEATs under-reports vocals, percussion and bass on a full mix, so those
    # three are never called phantom from the mix tags alone.
    for c in ("plucked_strings", "bowed_strings", "wind", "keys_synth"):
        if shares.get(c, 0) >= 15 and presence[c] < 10:
            flags.append("PHANTOM " + c)
    for c in truth_cats:
        if c in CATS and shares.get(c, 0) < 3 and "MISSED " + c not in flags:
            flags.append("MISSED(truth) " + c)
    if shares.get("folk_texture", 0) > 30:
        flags.append("CATCH-ALL folk_texture %.0f%%" % shares["folk_texture"])
    top = max(shares, key=shares.get)
    if shares[top] > 60 and top != "folk_texture":
        flags.append("DOMINANT %s %.0f%%" % (top, shares[top]))
    wanted = [c for c in CATS if presence[c] >= 30 or c in truth_cats]
    got = [c for c in wanted if shares.get(c, 0) >= 3]
    recall = 100.0 * len(got) / len(wanted) if wanted else 100.0
    return shares, flags, recall


def main():
    truth = json.load(open(TRUTH)) if os.path.exists(TRUTH) else {}
    lines = ["# Reference-song scorecard", "",
             "Truth = BEATs on the original mix (independent of the separator) + truth.json. "
             "Recall = share of categories present in the song that the model gave at least 3% of the energy.", ""]
    totals = {m: [] for m, _ in MODELS}
    for part, purpose in SONGS:
        path = find_song(part)
        if not path:
            continue
        key = "".join(c if c.isalnum() else "_" for c in part)[:30]
        tags = mix_tags(part, path)
        presence = tags["presence_pct"]
        truth_cats = truth.get(part, {}).get("present", [])
        lines += ["## " + os.path.basename(path), "purpose: " + purpose,
                  "audible in the mix (% of windows): " + ", ".join("%s %.0f" % (c, presence[c]) for c in CATS if presence[c] > 0),
                  "top mix labels: " + ", ".join("%s x%d" % tuple(kv) for kv in tags["top_labels"][:5]),
                  "from truth.json: " + (", ".join(truth_cats) or "(none given)"), ""]
        lines += ["| model | recall | " + " | ".join(CATS + ["folk_texture"]) + " |",
                  "|---|---|" + "---|" * (len(CATS) + 1)]
        notes = []
        for name, tag in MODELS:
            p = os.path.join(OUT_DIR, key + tag + ".json")
            if not os.path.exists(p):
                lines.append("| %s | not run |" % name)
                continue
            mj = json.load(open(p))
            shares, flags, recall = judge(mj, presence, truth_cats)
            totals[name].append((recall, len(flags)))
            lines.append("| %s | %.0f%% | " % (name, recall) + " | ".join("%.0f" % shares.get(c, 0) for c in CATS + ["folk_texture"]) + " |")
            notes.append("- " + name + ": " + ("; ".join(flags) if flags else "no flags"))
        lines += [""] + notes + [""]
    lines += ["## Overall", "", "| model | mean recall | mean flags per song | songs |", "|---|---|---|---|"]
    for name, _ in MODELS:
        t = totals[name]
        if t:
            lines.append("| %s | %.0f%% | %.1f | %d |" % (name, np.mean([r for r, _ in t]), np.mean([f for _, f in t]), len(t)))
    text = "\n".join(lines)
    open(os.path.join(OUT_DIR, "SCORECARD.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
