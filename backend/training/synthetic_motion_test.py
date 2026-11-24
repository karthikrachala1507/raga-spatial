# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - synthetic_motion_test.py
# Can the per-stem pan measurement tell a really moving instrument from a static one,
# once the instrument has been pulled out of a mix by the separator?
# Ground truth is built by hand: clean coherent stems (same song, same time) are panned
# in stereo; the target stem is either fixed at +0.4 or swept left-right (+-0.6, 8 s period);
# all others stay fixed. The stereo mix is separated (v5), the target stem's pan is measured
# with the same masks-on-left-and-right method as stem_stereo_motion.py, and compared with
# the pan that was applied.
# Output: data\evaluation\reference_songs\SYNTHETIC_MOTION.md
# Usage (from backend\):  python training\synthetic_motion_test.py

import os, glob, re, random
import numpy as np
import soundfile as sf
import soxr

from core import separator_8stem as sep
from core import real_motion
from training.analyze_reference_songs import OUT_DIR
from training.stem_stereo_motion import pan_series
from training.stereo_motion_check import SR

ALIGNED = paths.STEMS_ALIGNED_DIR
TARGETS = ["bowed_strings", "plucked_strings", "wind", "keys_synth"]
OTHERS = ["percussion", "bass"]
FIXED_PANS = {"vocals": 0.0, "percussion": -0.5, "bass": 0.0, "keys_synth": -0.4,
              "bowed_strings": -0.3, "plucked_strings": 0.5, "wind": -0.3}
N_PER = 12
NWIN = 1
SWING = 0.25
AMP = {"swept": 0.6, "small": 0.3}
random.seed(7)


def key_of(fn):
    m = re.match(r"(.+)__w(\d{4})\.wav$", os.path.basename(fn))
    return (m.group(1).rsplit("__", 1)[0], m.group(1), int(m.group(2))) if m else None


def index(cat):
    idx = {}
    for f in glob.glob(os.path.join(ALIGNED, cat, "*.wav")):
        k = key_of(f)
        if k:
            idx[(k[0].split("__")[0] + "__" + k[0].split("__")[1], k[2])] = f
    return idx


def load_run(idx, base, w0):
    parts = []
    for w in range(w0, w0 + NWIN):
        f = idx.get((base, w))
        if not f:
            return None
        a, _ = sf.read(f, dtype="float32", always_2d=True)
        parts.append(a.mean(axis=1))
    x = np.concatenate(parts)
    return soxr.resample(x, 16000, SR).astype(np.float32)


def pan_gains(p):
    ang = (np.asarray(p) + 1.0) * np.pi / 4.0
    return np.cos(ang), np.sin(ang)


skipped = {}


def main():
    dev = sep.pick_device()
    model, _ = sep.load_model(dev)
    cats = {c: index(c) for c in set(TARGETS + OTHERS)}
    lines = ["# Synthetic motion test", "",
             "Known pan applied to one stem (static +0.4, or swept +-0.6 over 6 s) inside a fixed-pan mix; "
             "then separated and measured. 'moving' = share of audible time with pan swing > %.2f within 4 s. "
             "'corr' = correlation of the measured pan with the applied pan (moving mixes only)." % SWING, "",
             "| target | mixes | static: moving | swept 0.6: moving | swept 0.3: moving | swept 0.6 corr | pan std static / 0.6 / 0.3 |",
             "|---|---|---|---|---|---|---|"]
    for tgt in TARGETS:
        bases = None
        for c in [tgt] + OTHERS:
            b = {k[0] for k in cats[c]}
            bases = b if bases is None else bases & b
        cand = []
        for base in sorted(bases):
            ws = sorted({k[1] for k in cats[tgt] if k[0] == base})
            for w0 in ws:
                if all((base, w) in cats[c] for c in [tgt] + OTHERS for w in range(w0, w0 + NWIN)):
                    cand.append((base, w0))
                    break
        random.shuffle(cand)
        res = {"static": [], "swept": [], "small": []}
        flag = {"static": [], "swept": [], "small": []}
        corrs = []
        used = 0
        for base, w0 in cand:
            if used >= N_PER:
                break
            stems = {c: load_run(cats[c], base, w0) for c in [tgt] + OTHERS}
            if any(v is None for v in stems.values()):
                continue
            n = min(len(v) for v in stems.values())
            stems = {c: v[:n] for c, v in stems.items()}
            t = np.arange(n) / SR
            for cond in ("static", "swept", "small"):
                L = np.zeros(n, np.float32)
                R = np.zeros(n, np.float32)
                for c, v in stems.items():
                    if c == tgt:
                        p = np.full(n, 0.4) if cond == "static" else AMP[cond] * np.sin(2 * np.pi * t / 6.0)
                    else:
                        p = np.full(n, FIXED_PANS[c])
                    gl, gr = pan_gains(p)
                    L += v * gl
                    R += v * gr
                peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
                L, R = L / peak * 0.8, R / peak * 0.8
                mid = (L + R) / 2
                est = sep.infer_16k(model, soxr.resample(mid, SR, sep.MODEL_SR).astype(np.float32), dev)
                sl, shares = sep.fullband_stems(L, est, SR)
                sr_, _ = sep.fullband_stems(R, est, SR)
                ci = model.categories.index(tgt)
                if shares[ci] < 0.03:
                    skipped[tgt] = skipped.get(tgt, 0) + 1
                    continue
                pan, en = pan_series(sl[ci], sr_[ci])
                loud = en > 0.05 * np.median(en + 1e-12)
                k = 4
                sw = np.array([pan[j:j + k].max() - pan[j:j + k].min() for j in range(len(pan) - k)
                               if loud[j:j + k].all()])
                if len(sw) == 0:
                    continue
                res[cond].append((float((sw > SWING).mean()), float(pan[loud].std())))
                a = real_motion.analyse_stem(sl[ci], sr_[ci], SR)
                flag[cond].append(bool(a and a["moving"]))
                if cond == "swept" and loud.sum() > 5:  # corr only for the big sweep
                    true = np.array([0.6 * np.sin(2 * np.pi * (i + 0.5) / 6.0) for i in range(len(pan))])
                    corrs.append(float(np.corrcoef(pan[loud], true[loud])[0, 1]))
            used += 1
        def avg(lst, i):
            return np.mean([x[i] for x in lst]) if lst else float("nan")
        lines.append("| %s | %d | %.0f%% | %.0f%% | %.0f%% | %.2f | %.2f / %.2f / %.2f |" % (
            tgt, len(res["swept"]), 100 * avg(res["static"], 0), 100 * avg(res["swept"], 0),
            100 * avg(res["small"], 0), float(np.mean(corrs)) if corrs else float("nan"),
            avg(res["static"], 1), avg(res["swept"], 1), avg(res["small"], 1)))
        lines.append("| &nbsp;&nbsp;app rule (real_motion) flags moving | | %.0f%% | %.0f%% | %.0f%% | | |" % tuple(100 * (np.mean(flag[c]) if flag[c] else float("nan")) for c in ("static", "swept", "small")))
        print(lines[-2], lines[-1], flush=True)
    text = "\n".join(lines)
    open(os.path.join(OUT_DIR, "SYNTHETIC_MOTION.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
