# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - make_synthetic_bass.py
# Generates electronic sub-bass stems (808-style): a sine that starts higher
# and drops, long decays, glides between notes, optional saturation, and
# sustained sub-sine lines. The training bass was real and synthesised bass
# GUITAR only, so songs built on an 808 ("Ordinary Person": 44% of its energy
# below 60 Hz) were not recognised as bass.
#
# Output: <project>\data\processed\stems_t2\bass\synth808__s<NNNN>__<clip>.wav
# 10 s, 16 kHz mono. Every source is one 60 s synthetic track with its own
# tempo, key, note pattern and tone, cut into 6 clips.
#
# Usage (from backend\):  python data_tools\make_synthetic_bass.py [number of tracks]
# Safe to rerun: existing tracks are skipped.

import os
import sys
import numpy as np
import soundfile as sf

SR      = 16000
TRACK_S = 60
CLIPS   = 6
OUT_DIR = os.path.join(paths.STEMS_T2_DIR, "bass")


def note_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def render_track(rng):
    n = SR * TRACK_S
    out = np.zeros(n, dtype=np.float64)
    bpm = rng.uniform(65, 160)
    step = 60.0 / bpm / 4                              # one 16th note, in seconds
    root = rng.integers(26, 41)                        # about 37-82 Hz
    scales = [[0, 3, 5, 7, 10], [0, 2, 3, 5, 7, 8, 10], [0, 2, 4, 7, 9], [0, 1, 4, 5, 7, 8, 11]]
    scale = scales[int(rng.integers(len(scales)))]
    style = rng.choice(["808", "808", "808", "sustained", "short"])
    drop = rng.uniform(0.2, 1.2)                       # how far the pitch falls after the hit
    drop_tau = rng.uniform(0.015, 0.08)
    tau = {"808": rng.uniform(0.3, 1.6), "sustained": 4.0, "short": rng.uniform(0.08, 0.3)}[style]
    drive = rng.choice([0.0, 0.0, 1.5, 3.0, 6.0])      # 0 = clean sine
    glide_p = 0.2 if style == "808" else 0.05
    density = rng.uniform(0.15, 0.6)
    pattern = rng.random(16) < density
    pattern[0] = True
    accent = rng.uniform(0.6, 1.0, 16)
    bar_notes = [int(rng.choice(scale)) for _ in range(16)]
    phase = 0.0
    t_grid = np.arange(int(step * SR * 16))
    n_steps = int(TRACK_S / step)
    onsets = [i for i in range(n_steps) if pattern[i % 16]
              and not (rng.random() < 0.08)]           # a few dropped notes
    for k, i in enumerate(onsets):
        start = int(i * step * SR)
        nxt = int(onsets[k + 1] * step * SR) if k + 1 < len(onsets) else n
        length = min(nxt - start, int(SR * (4.0 if style == "sustained" else 2.5)))
        if length < 200:
            continue
        t = np.arange(length) / SR
        f0 = note_freq(root + bar_notes[i % 16])
        f_target = f0
        if rng.random() < glide_p and k + 1 < len(onsets):
            f_target = note_freq(root + bar_notes[onsets[k + 1] % 16])
        glide = np.clip(t / max(length / SR, 1e-3), 0, 1) ** 2
        freq = (f0 + (f_target - f0) * glide) * (1.0 + drop * np.exp(-t / drop_tau))
        ph = 2 * np.pi * np.cumsum(freq) / SR
        wave = np.sin(ph)
        if drive > 0:
            wave = np.tanh(drive * wave) / np.tanh(drive)
        attack = np.minimum(t / 0.003, 1.0)
        release = np.minimum((length / SR - t) / 0.04, 1.0).clip(0, 1)
        env = attack * release * np.exp(-t / tau)
        out[start:start + length] += wave * env * accent[i % 16]
    peak = np.abs(out).max()
    return (out / peak * 0.9).astype(np.float32) if peak > 1e-6 else None


def main():
    n_tracks = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    os.makedirs(OUT_DIR, exist_ok=True)
    made = 0
    for s in range(n_tracks):
        first = os.path.join(OUT_DIR, "synth808__s%04d__%04d.wav" % (s, 0))
        if os.path.exists(first):
            continue
        rng = np.random.default_rng(10_000 + s)        # same track every time for the same s
        track = render_track(rng)
        if track is None:
            continue
        for c in range(CLIPS):
            clip = track[c * SR * 10:(c + 1) * SR * 10]
            if np.sqrt(np.mean(clip ** 2)) < 0.02:     # skip near-silent clips
                continue
            name = "synth808__s%04d__%04d.wav" % (s, c)
            tmp = os.path.join(OUT_DIR, name + ".tmp")
            sf.write(tmp, clip, SR, format="WAV", subtype="PCM_16")
            os.replace(tmp, os.path.join(OUT_DIR, name))
        made += 1
    print("synthetic bass tracks made: %d (of %d requested); clips on disk: %d" % (
        made, n_tracks, len([f for f in os.listdir(OUT_DIR) if f.startswith("synth808__")])))


if __name__ == "__main__":
    main()
