# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - stereo_motion_check.py
# Does the original song really move anything across the stereo field?
# Before the app moves a stem, there must be motion to follow. This measures it
# directly in the stereo mix: for each frequency band, the left/right balance
# (pan, -1 = hard left, +1 = hard right) every second, and how much it changes.
#
#   steady   the balance barely moves: nothing is panned over time
#   movement the balance in a band swings by a large amount within a few seconds
#
# Output: data\evaluation\reference_songs\STEREO_MOTION.md
# Usage (from backend\):  python training\stereo_motion_check.py

import os
import subprocess
import numpy as np

from training.analyze_reference_songs import SONGS, OUT_DIR, find_song, ffmpeg_exe

SR = 22050
BANDS = [("low 80-300 Hz", 80, 300), ("mid 300-2k", 300, 2000), ("high 2k-8k", 2000, 8000)]
WIN_SEC = 1.0
SWING = 0.25          # pan change within 4 s that counts as movement


def load_stereo(path):
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T


def band_pan(stereo, lo, hi):
    n_fft, hop = 2048, 1024
    win = np.hanning(n_fft)
    def spec(x):
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
        return np.abs(np.fft.rfft(frames, axis=1))
    L, R = spec(stereo[0]), spec(stereo[1])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    sel = (freqs >= lo) & (freqs < hi)
    l = (L[:, sel] ** 2).sum(axis=1)
    r = (R[:, sel] ** 2).sum(axis=1)
    per = int(WIN_SEC * SR / hop)
    n = len(l) // per
    l, r = l[:n * per].reshape(n, per).sum(axis=1), r[:n * per].reshape(n, per).sum(axis=1)
    return (r - l) / (r + l + 1e-12), (r + l)


def main():
    lines = ["# Stereo motion check", "",
             "Pan per second in three bands (-1 left, +1 right). 'swing' = largest pan change inside any 4 s; "
             "'moving seconds' = share of the song where it changes by more than %.2f within 4 s." % SWING, ""]
    lines += ["| song | band | mean pan | pan spread (std) | biggest swing | moving seconds |", "|---|---|---|---|---|---|"]
    for part, purpose in SONGS:
        path = find_song(part)
        if not path:
            continue
        st = load_stereo(path)
        name = os.path.basename(path)[:34]
        for label, lo, hi in BANDS:
            pan, energy = band_pan(st, lo, hi)
            loud = energy > 0.05 * np.median(energy + 1e-12)
            if loud.sum() < 8:
                lines.append("| %s | %s | silent | | | |" % (name, label))
                continue
            k = 4
            swing = np.array([pan[i:i + k].max() - pan[i:i + k].min() for i in range(len(pan) - k)])
            ok = np.array([loud[i:i + k].all() for i in range(len(pan) - k)])
            sw = swing[ok] if ok.any() else np.zeros(1)
            lines.append("| %s | %s | %+.2f | %.2f | %.2f | %.0f%% |" % (
                name, label, float(pan[loud].mean()), float(pan[loud].std()),
                float(sw.max()), 100.0 * float((sw > SWING).mean())))
    text = "\n".join(lines)
    open(os.path.join(OUT_DIR, "STEREO_MOTION.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
