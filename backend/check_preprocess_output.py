# Run on Karthik's machine to check exact preprocess output structure
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from core.preprocess import (load_audio, normalize_audio, extract_mel_spectrogram,
                              separate_harmonic_percussive, detect_onsets,
                              extract_chroma, create_windows, resample_for_beats,
                              get_tempo_and_beats)

audio, sr = load_audio("test.flac")
audio     = normalize_audio(audio)

mel                  = extract_mel_spectrogram(audio, sr)
harmonic, percussive = separate_harmonic_percussive(audio)
onsets               = detect_onsets(audio, sr)
chroma               = extract_chroma(audio, sr)
tempo, beats         = get_tempo_and_beats(audio, sr)

# Try both call signatures for create_windows
try:
    short_w, long_w = create_windows(audio, sr)
    print("create_windows signature: (audio, sr)")
except TypeError:
    try:
        short_w = create_windows(audio, sr, window_sec=3.0)
        long_w  = create_windows(audio, sr, window_sec=12.0)
        print("create_windows signature: (audio, sr, window_sec=)")
    except TypeError:
        short_w, long_w = [], []
        print("create_windows signature: unknown - check preprocess.py manually")

print()
print("=== SHAPES ===")
print("mel shape:          ", mel.shape)
print("harmonic shape:     ", harmonic.shape)
print("percussive shape:   ", percussive.shape)
print("chroma shape:       ", chroma.shape)
print("sr:                 ", sr)
print("audio duration:     ", round(len(audio)/sr, 2), "s")
print("chroma frames/sec:  ", round(chroma.shape[1] / (len(audio)/sr), 2))
print("chroma hop sec:     ", round(len(audio)/sr / chroma.shape[1], 4))
print()
print("=== ONSETS ===")
print("onsets type:        ", type(onsets))
print("onsets count:       ", len(onsets))
print("onsets[:8]:         ", onsets[:8])
print("onset unit:         seconds (float)")
print()
print("=== BEATS ===")
print("tempo:              ", round(float(tempo), 2), "BPM")
print("beats type:         ", type(beats))
print("beats count:        ", len(beats))
print("beats[:8]:          ", beats[:8])
print("beats unit:         frames (int) - multiply by hop/sr for seconds")
print()
print("=== WINDOWS ===")
if short_w:
    print("short windows count:", len(short_w))
    print("short window shape: ", short_w[0].shape if hasattr(short_w[0], 'shape') else type(short_w[0]))
if long_w:
    print("long windows count: ", len(long_w))
    print("long window shape:  ", long_w[0].shape if hasattr(long_w[0], 'shape') else type(long_w[0]))
