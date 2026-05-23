# Run on Karthik's machine to diagnose why motion detection returns 0 events
# Place in backend/ and run: python debug_motion.py

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from core.preprocess import (load_audio, normalize_audio, extract_chroma,
                              detect_onsets, get_tempo_and_beats)
from scipy.ndimage import uniform_filter1d

audio, sr = load_audio("test.flac")
audio     = normalize_audio(audio)
chroma    = extract_chroma(audio, sr)
onsets    = detect_onsets(audio, sr)
tempo, beats = get_tempo_and_beats(audio, sr)

chroma_hop_sec = 512 / sr   # 0.0116s per frame
N = chroma.shape[1]
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Smooth and get dominant pitch per frame
chroma_smooth  = uniform_filter1d(chroma, size=3, axis=1)
dominant_pitch = np.argmax(chroma_smooth, axis=0)

print("=== DOMINANT PITCH EVERY 0.5s ===")
step = int(0.5 / chroma_hop_sec)
for i in range(0, min(N, int(30/chroma_hop_sec)), step):
    t     = round(i * chroma_hop_sec, 2)
    note  = NOTE_NAMES[dominant_pitch[i]]
    # energy of dominant pitch at this frame
    energy = round(float(chroma_smooth[dominant_pitch[i], i]), 3)
    print(str(t).ljust(6) + "s  " + note.ljust(3) + "  energy=" + str(energy))

print()
print("=== PITCH CHANGES (semitone jumps) ===")
changes = []
for i in range(1, min(N, int(30/chroma_hop_sec))):
    curr = int(dominant_pitch[i])
    prev = int(dominant_pitch[i-1])
    diff = (curr - prev) % 12
    if diff > 6:
        diff = diff - 12
    if diff != 0:
        t = round(i * chroma_hop_sec, 2)
        changes.append((t, NOTE_NAMES[prev], NOTE_NAMES[curr], diff))

print("Total pitch changes in 30s: " + str(len(changes)))
print("First 30 changes:")
for t, frm, to, diff in changes[:30]:
    direction = "UP" if diff > 0 else "DOWN"
    print(str(t).ljust(6) + "s  " + frm + " -> " + to + "  (" + direction + " " + str(abs(diff)) + " semitones)")

print()
print("=== JUMP SIZE DISTRIBUTION ===")
jump_sizes = [abs(c[3]) for c in changes]
for size in range(1, 13):
    count = jump_sizes.count(size)
    bar   = "#" * count
    print(str(size).ljust(3) + " semitones: " + bar + " " + str(count))

print()
print("=== ONSET DENSITY PER 5s SECTION ===")
for sec in range(0, 30, 5):
    mask    = (onsets >= sec) & (onsets < sec + 5)
    count   = int(np.sum(mask))
    density = round(count / 5.0, 2)
    bar     = "#" * count
    print(str(sec) + "-" + str(sec+5) + "s: " + bar + " " + str(count) + " onsets (" + str(density) + "/s)")

print()
print("=== CONSECUTIVE SAME-PITCH RUN LENGTHS ===")
runs = []
run_start = 0
for i in range(1, N):
    if dominant_pitch[i] != dominant_pitch[i-1]:
        run_len = i - run_start
        if run_len >= 5:
            runs.append((round(run_start * chroma_hop_sec, 2),
                         round(i * chroma_hop_sec, 2),
                         NOTE_NAMES[dominant_pitch[run_start]],
                         run_len))
        run_start = i
print("Runs >= 5 frames (>= 0.06s) on same pitch:")
for r in runs[:20]:
    print(str(r[0]) + "s-" + str(r[1]) + "s  note=" + r[2] + "  frames=" + str(r[3]))
