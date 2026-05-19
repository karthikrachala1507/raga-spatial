"""
Raga Spatial — core/preprocess.py
Audio Preprocessing Module

What this does:
- Loads any MP3/WAV/FLAC audio file
- Normalizes loudness
- Resamples to standard 44100 Hz
- Extracts mel spectrograms
- Detects onsets (note beginnings)
- Extracts harmonic structures
- Prepares sliding windows for BEATs detection
"""

import numpy as np
import librosa
import soundfile as sf
import os


# ── Constants ──────────────────────────────────────────────────────────────
TARGET_SR = 44100        # Standard sample rate for all processing
TARGET_SR_BEATS = 16000  # BEATs model requires 16kHz input
N_MELS = 128             # Mel spectrogram bands
HOP_LENGTH = 512         # Hop size for STFT
N_FFT = 2048             # FFT window size
SHORT_WINDOW = 3.0       # Seconds — for rapid instrument changes
LONG_WINDOW = 12.0       # Seconds — for motion event analysis


# ══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION — call this first in the pipeline
# ══════════════════════════════════════════════════════════════════════════
def preprocess_audio(file_path: str) -> dict:
    """
    Full preprocessing pipeline for one audio file.
    Returns a dict with everything downstream modules need.

    Usage:
        result = preprocess_audio("outputs/uploads/myjob.mp3")
        audio = result["audio"]
        sr = result["sr"]
        mel = result["mel_spectrogram"]
    """
    print(f"[Preprocess] Loading: {file_path}")

    # ── Step 1: Load audio ─────────────────────────────────────────────
    audio, sr = load_audio(file_path)
    print(f"[Preprocess] Loaded — SR: {sr}Hz, Duration: {len(audio)/sr:.2f}s, Shape: {audio.shape}")

    # ── Step 2: Normalize ──────────────────────────────────────────────
    audio = normalize_audio(audio)
    print(f"[Preprocess] Normalized — Peak: {np.max(np.abs(audio)):.4f}")

    # ── Step 3: Get duration info ──────────────────────────────────────
    duration = len(audio) / sr
    print(f"[Preprocess] Duration: {duration:.2f} seconds")

    # ── Step 4: Extract mel spectrogram ───────────────────────────────
    mel_spec = extract_mel_spectrogram(audio, sr)
    print(f"[Preprocess] Mel spectrogram shape: {mel_spec.shape}")

    # ── Step 5: Extract harmonic + percussive components ──────────────
    harmonic, percussive = separate_harmonic_percussive(audio)
    print(f"[Preprocess] Harmonic/Percussive separation done")

    # ── Step 6: Detect onsets ─────────────────────────────────────────
    onset_times = detect_onsets(audio, sr)
    print(f"[Preprocess] Onsets detected: {len(onset_times)}")

    # ── Step 7: Extract chroma (pitch/key info) ────────────────────────
    chroma = extract_chroma(harmonic, sr)
    print(f"[Preprocess] Chroma shape: {chroma.shape}")

    # ── Step 8: Create sliding windows ────────────────────────────────
    short_windows = create_windows(audio, sr, window_sec=SHORT_WINDOW)
    long_windows = create_windows(audio, sr, window_sec=LONG_WINDOW)
    print(f"[Preprocess] Short windows ({SHORT_WINDOW}s): {len(short_windows)}")
    print(f"[Preprocess] Long windows ({LONG_WINDOW}s): {len(long_windows)}")

    # ── Step 9: Resample for BEATs model ──────────────────────────────
    audio_16k = resample_for_beats(audio, sr)
    print(f"[Preprocess] Resampled to 16kHz for BEATs — Shape: {audio_16k.shape}")

    print(f"[Preprocess] ✓ Complete")

    return {
        # Raw audio
        "audio": audio,                    # Full normalized audio (44100 Hz)
        "audio_16k": audio_16k,            # Resampled for BEATs (16000 Hz)
        "sr": sr,                          # Sample rate (44100)
        "duration": duration,              # Duration in seconds

        # Spectral features
        "mel_spectrogram": mel_spec,       # (n_mels, time_frames)
        "chroma": chroma,                  # (12, time_frames) — pitch classes

        # Separated components
        "harmonic": harmonic,              # Harmonic content only
        "percussive": percussive,          # Percussive content only

        # Event info
        "onset_times": onset_times,        # List of onset timestamps (seconds)

        # Sliding windows for BEATs
        "short_windows": short_windows,    # List of {start, end, audio} dicts
        "long_windows": long_windows,      # List of {start, end, audio} dicts
    }


# ══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def load_audio(file_path: str) -> tuple:
    """
    Load audio file in any format (MP3, WAV, FLAC, AAC).
    Always returns mono audio at TARGET_SR (44100 Hz).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # librosa handles MP3, WAV, FLAC, AAC automatically
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio so peak amplitude = 1.0
    Prevents clipping and ensures consistent model input.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def extract_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert audio waveform to mel spectrogram.
    This is the primary feature representation for AI models.

    Returns: np.ndarray shape (N_MELS, time_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=8000  # Max frequency — covers all Indian instruments
    )
    # Convert to log scale (dB) — more perceptually meaningful
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def separate_harmonic_percussive(audio: np.ndarray) -> tuple:
    """
    Separate audio into harmonic (melodic) and percussive components.
    Harmonic → feeds pitch/chroma analysis and BEATs
    Percussive → feeds rhythm/onset analysis (Mridangam, Tabla detection)
    """
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic, percussive


def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect note onset times in seconds.
    Used by motion engine to track melodic progressions (C→D→E→F).
    """
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=HOP_LENGTH,
        backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    return onset_times


def extract_chroma(harmonic: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract chroma features — represents 12 pitch classes (C,C#,D,...,B).
    Used by motion engine to detect side-by-side key progressions.

    Returns: np.ndarray shape (12, time_frames)
    """
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=12
    )
    return chroma


def create_windows(audio: np.ndarray, sr: int, window_sec: float) -> list:
    """
    Slice full audio into overlapping windows for BEATs detection.
    50% overlap between windows ensures no events are missed at boundaries.

    Returns list of dicts:
    [
        {"start": 0.0, "end": 3.0, "audio": np.array(...)},
        {"start": 1.5, "end": 4.5, "audio": np.array(...)},
        ...
    ]
    """
    window_samples = int(window_sec * sr)
    hop_samples = window_samples // 2  # 50% overlap
    windows = []

    start_sample = 0
    while start_sample + window_samples <= len(audio):
        end_sample = start_sample + window_samples
        window_audio = audio[start_sample:end_sample]

        windows.append({
            "start": start_sample / sr,           # Start time in seconds
            "end": end_sample / sr,               # End time in seconds
            "audio": window_audio,                # Raw audio chunk
            "audio_16k": resample_for_beats(window_audio, sr)  # 16kHz for BEATs
        })

        start_sample += hop_samples

    # Handle last partial window if song doesn't divide evenly
    if start_sample < len(audio):
        remaining = audio[start_sample:]
        # Pad with zeros to fill window
        padded = np.zeros(window_samples)
        padded[:len(remaining)] = remaining
        windows.append({
            "start": start_sample / sr,
            "end": len(audio) / sr,
            "audio": padded,
            "audio_16k": resample_for_beats(padded, sr)
        })

    return windows


def resample_for_beats(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Resample audio from 44100 Hz to 16000 Hz.
    BEATs transformer model requires 16kHz input.
    """
    if sr == TARGET_SR_BEATS:
        return audio
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR_BEATS)
    return audio_16k


def get_tempo_and_beats(audio: np.ndarray, sr: int) -> tuple:
    """
    Detect song tempo (BPM) and beat positions.
    Used for rhythm-aware motion triggering.
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=audio,
        sr=sr,
        hop_length=HOP_LENGTH
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    return float(tempo), beat_times


def save_audio(audio: np.ndarray, sr: int, output_path: str):
    """
    Save processed audio to WAV file.
    Used to save stems and final spatial output.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, sr)
    print(f"[Preprocess] Saved: {output_path}")
