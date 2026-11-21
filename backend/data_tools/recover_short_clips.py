# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - recover_short_clips.py
# The first clip cutter (verify_stems.py) needed a full 10 s window, so every
# raw file shorter than 10 s was thrown away: about 400 files, mostly
# FreeSounds one-shots and short phrases. This tool recovers them: each file of
# 2 to 10 s becomes one 10 s clip (the rest is silence, as in training).
#
# Usage (from backend\):  python data_tools\recover_short_clips.py
# Safe to rerun: clips already recovered are skipped.

import os
import numpy as np
import soxr
import soundfile as sf

SR, LEN = 16000, 160000
MIN_RMS = 0.005


def main():
    total = {}
    for cat in sorted(os.listdir(paths.RAW_STEMS_DIR)):
        src = os.path.join(paths.RAW_STEMS_DIR, cat)
        dst = os.path.join(paths.STEMS_T1_DIR, cat)
        if not os.path.isdir(src) or not os.path.isdir(dst):
            continue
        n = 0
        for i, f in enumerate(sorted(os.listdir(src))):
            if not f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                continue
            out = os.path.join(dst, "%s_s%05d_000.wav" % (cat, i))
            if os.path.exists(out):
                continue
            try:
                info = sf.info(os.path.join(src, f))
                if not (2 <= info.duration < 10):
                    continue                  # long files were already cut; skip decoding them
                a, rate = sf.read(os.path.join(src, f), dtype="float32", always_2d=True)
                a = soxr.resample(a.mean(axis=1), rate, SR).astype(np.float32)
            except Exception:
                continue
            if not (2 * SR <= len(a) < LEN):
                continue                      # long files were already cut
            if float(np.sqrt(np.mean(a ** 2))) < MIN_RMS:
                continue
            a = a / (np.abs(a).max() + 1e-8) * 0.9
            sf.write(out + ".tmp", np.pad(a, (0, LEN - len(a))), SR, format="WAV", subtype="PCM_16")
            os.replace(out + ".tmp", out)
            n += 1
        total[cat] = n
    print("recovered clips:", total, "| total", sum(total.values()))


if __name__ == "__main__":
    main()
