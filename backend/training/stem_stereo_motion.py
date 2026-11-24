# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - stem_stereo_motion.py
# Which separated instrument really moves across the stereo field in the ORIGINAL song?
# The separator works on the mono mix, and its masks do not depend on the signal they are
# applied to, so the same masks are applied to the left and the right channel separately.
# Then, for every stem, the pan (-1 left .. +1 right) is measured once a second.
# A stem "moves" when its own pan swings by more than SWING inside 4 s while it is audible.
# Output: data\evaluation\reference_songs\STEM_MOTION.md
# Usage (from backend\):  python training\stem_stereo_motion.py [song-name-part ...]

import os
import sys
import numpy as np
import soxr

from core import separator_8stem as sep
from training.analyze_reference_songs import SONGS, OUT_DIR, find_song
from training.stereo_motion_check import load_stereo, SR

SWING = 0.25
MIN_SHARE = 0.03


def pan_series(l, r, n_fft=2048, hop=1024):
    win = np.hanning(n_fft)
    def power(x):
        fr = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
        return (np.abs(np.fft.rfft(fr, axis=1)) ** 2).sum(axis=1)
    pl, pr = power(l), power(r)
    per = int(SR / hop)
    n = len(pl) // per
    pl, pr = pl[:n * per].reshape(n, per).sum(1), pr[:n * per].reshape(n, per).sum(1)
    return (pr - pl) / (pr + pl + 1e-12), pr + pl


def main():
    only = [a.lower() for a in sys.argv[1:]]
    dev = sep.pick_device()
    model, _ = sep.load_model(dev)
    lines = ["# Per-stem stereo motion", "",
             "Masks from the separator applied to left and right separately; pan per second. "
             "'moving' = share of audible time where the stem's own pan changes by more than %.2f within 4 s." % SWING,
             "", "| song | stem | share | mean pan | pan std | moving |", "|---|---|---|---|---|---|"]
    for part, _ in SONGS:
        if only and not any(o in part.lower() for o in only):
            continue
        path = find_song(part)
        if not path:
            continue
        st = load_stereo(path)                       # (2, n) at SR
        mid = st.mean(axis=0)
        est = sep.infer_16k(model, soxr.resample(mid, SR, sep.MODEL_SR).astype(np.float32), dev)
        stems_l, shares = sep.fullband_stems(st[0], est, SR)
        stems_r, _ = sep.fullband_stems(st[1], est, SR)
        name = os.path.basename(path)[:30]
        for i, cat in enumerate(model.categories):
            if shares[i] < MIN_SHARE:
                continue
            pan, en = pan_series(stems_l[i], stems_r[i])
            loud = en > 0.05 * np.median(en + 1e-12)
            if loud.sum() < 8:
                continue
            k = 4
            sw = np.array([pan[j:j + k].max() - pan[j:j + k].min() for j in range(len(pan) - k)
                           if loud[j:j + k].all()])
            moving = 100.0 * float((sw > SWING).mean()) if len(sw) else 0.0
            lines.append("| %s | %s | %.0f%% | %+.2f | %.2f | %.0f%% |" % (
                name, cat, shares[i] * 100, float(pan[loud].mean()), float(pan[loud].std()), moving))
        print(name, "done", flush=True)
    text = "\n".join(lines)
    open(os.path.join(OUT_DIR, "STEM_MOTION.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
