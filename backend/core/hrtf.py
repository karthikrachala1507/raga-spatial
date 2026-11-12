# Raga Spatial - core/hrtf.py
# Phase 8: HRTF Binaural Rendering

import numpy as np
import os
import soundfile as sf
from scipy.signal import fftconvolve

MODELS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SOFA_FILE    = os.path.join(MODELS_DIR, "mit_kemar_normal_pinna.sofa")
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "spatial")
TARGET_SR    = 44100


def load_sofa(sofa_path=None):
    if sofa_path is None:
        sofa_path = SOFA_FILE
    if not os.path.exists(sofa_path):
        raise FileNotFoundError("SOFA file not found: " + sofa_path)

    print("[HRTF] Loading SOFA file: " + sofa_path)
    try:
        import pysofaconventions as pysofa
        sofa      = pysofa.SOFAFile(sofa_path, "r")
        positions = sofa.getVariableValue("SourcePosition")
        data      = sofa.getVariableValue("Data.IR")
        sr        = int(sofa.getVariableValue("Data.SamplingRate").item())
        sofa.close()

        azimuths   = positions[:, 0].astype(np.float32)
        elevations = positions[:, 1].astype(np.float32)
        hrir_left  = data[:, 0, :].astype(np.float32)
        hrir_right = data[:, 1, :].astype(np.float32)

        print("[HRTF] Loaded " + str(len(azimuths)) + " HRIRs"
              + " | SR: " + str(sr) + "Hz"
              + " | Taps: " + str(hrir_left.shape[1]))

        return {
            "azimuths":   azimuths,
            "elevations": elevations,
            "hrir_left":  hrir_left,
            "hrir_right": hrir_right,
            "sr":         sr,
            "n_taps":     hrir_left.shape[1],
        }
    except Exception as e:
        raise RuntimeError("Failed to load SOFA file: " + str(e))


def get_hrir_for_azimuth(sofa_data, azimuth_deg, elevation_deg=0.0):
    # Our convention:  0=front, 90=right, 180=rear, 270=left
    # MIT KEMAR:       0=front, 90=left,  180=rear, 270=right
    mit_az     = (360.0 - azimuth_deg) % 360.0
    azimuths   = sofa_data["azimuths"]
    elevations = sofa_data["elevations"]

    elev_diffs = np.abs(elevations - elevation_deg)
    min_elev   = np.min(elev_diffs)
    elev_mask  = elev_diffs <= min_elev + 1.0

    filtered_az  = azimuths[elev_mask]
    filtered_idx = np.where(elev_mask)[0]

    az_diffs = np.abs(filtered_az - mit_az)
    az_diffs = np.minimum(az_diffs, 360.0 - az_diffs)
    closest  = filtered_idx[np.argmin(az_diffs)]

    return sofa_data["hrir_left"][closest], sofa_data["hrir_right"][closest]


def spatialise_stem(audio_mono, azimuth_deg, distance, sofa_data, stem_sr=44100):
    hrir_l, hrir_r = get_hrir_for_azimuth(sofa_data, azimuth_deg)

    if sofa_data["sr"] != stem_sr:
        from scipy.signal import resample
        target_len = int(len(hrir_l) * stem_sr / sofa_data["sr"])
        hrir_l = resample(hrir_l, target_len)
        hrir_r = resample(hrir_r, target_len)

    left  = fftconvolve(audio_mono, hrir_l, mode="full")[:len(audio_mono)]
    right = fftconvolve(audio_mono, hrir_r, mode="full")[:len(audio_mono)]

    gain   = 1.0 / (1.0 + distance * 3.0)
    left  *= gain
    right *= gain

    return np.stack([left, right], axis=0)


# ── Trajectory rendering (overlap-add, no gaps between blocks) ───────────────

AZ_STEP_DEG = 5.0      # KEMAR horizontal resolution
OLA_BLOCK   = 2048     # ~46 ms at 44.1 kHz
OLA_HOP     = 1024     # 50% overlap: periodic Hann windows sum to exactly 1


def _prepared_hrir(sofa_data, azimuth_deg, sr):
    hrir_l, hrir_r = get_hrir_for_azimuth(sofa_data, azimuth_deg)
    if sofa_data["sr"] != sr:
        from scipy.signal import resample
        target_len = int(len(hrir_l) * sr / sofa_data["sr"])
        hrir_l = resample(hrir_l, target_len)
        hrir_r = resample(hrir_r, target_len)
    return hrir_l, hrir_r


def spatialise_static(audio_mono, azimuth_deg, sofa_data, sr=44100):
    """One HRTF convolution for a source that does not move. No gain applied."""
    hrir_l, hrir_r = _prepared_hrir(sofa_data, azimuth_deg, sr)
    n = len(audio_mono)
    left  = fftconvolve(audio_mono, hrir_l, mode="full")[:n]
    right = fftconvolve(audio_mono, hrir_r, mode="full")[:n]
    return np.stack([left, right], axis=0).astype(np.float32)


def spatialise_trajectory(audio_mono, azimuths, resolution_sec, sofa_data, sr=44100):
    """
    HRTF rendering for a source whose azimuth changes over time.

    azimuths[i] is the azimuth at time i * resolution_sec. The signal is split
    into 50%-overlapping Hann-windowed blocks; each block is convolved with the
    HRIR of its own azimuth and the results are overlap-added with full HRIR
    tails. The windows sum to 1, so there is no amplitude ripple between blocks
    (the old per-chunk fade-in/fade-out produced a 10 Hz tremolo).
    """
    n = len(audio_mono)
    azimuths = np.asarray(azimuths, dtype=np.float32)
    if n == 0:
        return np.zeros((2, 0), dtype=np.float32)
    if len(azimuths) == 0:
        return spatialise_static(audio_mono, 0.0, sofa_data, sr)

    window   = np.hanning(OLA_BLOCK + 1)[:-1].astype(np.float32)   # periodic Hann
    starts   = np.arange(-OLA_HOP, n, OLA_HOP)
    centers  = (starts + OLA_BLOCK / 2) / sr
    idx      = np.clip((centers / resolution_sec).astype(int), 0, len(azimuths) - 1)
    block_az = (np.round(azimuths[idx] / AZ_STEP_DEG) * AZ_STEP_DEG) % 360.0

    unique_az = np.unique(block_az)
    if len(unique_az) == 1:
        return spatialise_static(audio_mono, float(unique_az[0]), sofa_data, sr)

    taps = _prepared_hrir(sofa_data, 0.0, sr)[0].shape[0]
    out  = np.zeros((2, n + taps), dtype=np.float32)

    for az in unique_az:
        weight = np.zeros(n, dtype=np.float32)
        for s in starts[block_az == az]:
            a, b = max(s, 0), min(s + OLA_BLOCK, n)
            if b > a:
                weight[a:b] += window[a - s:b - s]
        nz = np.flatnonzero(weight)
        if len(nz) == 0:
            continue
        a, b = int(nz[0]), int(nz[-1]) + 1
        segment = audio_mono[a:b] * weight[a:b]
        hrir_l, hrir_r = _prepared_hrir(sofa_data, float(az), sr)
        left  = fftconvolve(segment, hrir_l, mode="full")
        right = fftconvolve(segment, hrir_r, mode="full")
        out[0, a:a + len(left)]  += left
        out[1, a:a + len(right)] += right

    return out[:, :n]


# ── Distance effects (depth layering) ────────────────────────────────────────

def distance_params(distance):
    """Gain, low-pass cutoff (Hz) and reverb mix for a distance in 0..1."""
    d = float(min(max(distance, 0.0), 1.0))
    return {
        "gain":       1.0 / (1.0 + d * 4.0),
        "cutoff_hz":  18000.0 - d * 12000.0,
        "reverb_mix": d * 0.3,
    }


def apply_distance_effect(audio_mono, distance, sr=44100, lowpass=True):
    """Dry part of the distance cue: level drop and high-frequency roll-off."""
    p = distance_params(distance)
    out = audio_mono
    if lowpass and p["cutoff_hz"] < 0.45 * sr:
        from scipy.signal import butter, sosfilt
        sos = butter(2, p["cutoff_hz"], btype="low", fs=sr, output="sos")
        out = sosfilt(sos, out)
    return (out * p["gain"] * (1.0 - 0.5 * p["reverb_mix"])).astype(np.float32)


def _room_impulse(distance, sr, seed):
    """Synthetic stereo room tail: decaying noise, longer and later when farther."""
    from scipy.signal import butter, sosfilt
    d      = float(min(max(distance, 0.0), 1.0))
    rt60   = 0.9 + 0.8 * d
    length = int(rt60 * sr)
    rng    = np.random.RandomState(seed)
    t      = np.arange(length) / sr
    tail   = rng.randn(2, length).astype(np.float32) * np.exp(-6.91 * t / rt60)
    sos    = butter(2, [250.0, 5000.0], btype="band", fs=sr, output="sos")
    tail   = sosfilt(sos, tail, axis=1)
    tail  /= np.sqrt(np.sum(tail ** 2, axis=1, keepdims=True)) + 1e-9
    delay  = int((0.012 + 0.02 * d) * sr)
    return np.concatenate([np.zeros((2, delay), dtype=np.float32),
                           tail.astype(np.float32)], axis=1)


def reverb_tail(audio_mono, distance, sr=44100, seed=0):
    """Wet part of the distance cue: a diffuse stereo tail, louder when farther."""
    p = distance_params(distance)
    n = len(audio_mono)
    if p["reverb_mix"] <= 0.0 or n == 0:
        return np.zeros((2, n), dtype=np.float32)
    ir   = _room_impulse(distance, sr, seed)
    left  = fftconvolve(audio_mono, ir[0], mode="full")[:n]
    right = fftconvolve(audio_mono, ir[1], mode="full")[:n]
    wet   = np.stack([left, right], axis=0) * p["gain"] * p["reverb_mix"]
    return wet.astype(np.float32)


def spatialise_stem_with_motion(audio_mono, spatial_timeline,
                                 stem_label, sofa_data, sr=44100):
    """2-stem pipeline: follow the label's azimuth/distance through the timeline."""
    n_samples = len(audio_mono)
    if len(spatial_timeline) < 2:
        return spatialise_stem(audio_mono, 0.0, 0.5, sofa_data, sr)
    resolution = spatial_timeline[1]["time_sec"] - spatial_timeline[0]["time_sec"]

    azimuths  = np.zeros(len(spatial_timeline), dtype=np.float32)
    distances = np.full(len(spatial_timeline), 0.5, dtype=np.float32)
    for i, entry in enumerate(spatial_timeline):
        for instr in entry.get("instruments", []):
            if (instr.get("raw_label") == stem_label or
                    instr.get("label") == stem_label):
                azimuths[i]  = instr["azimuth"]
                distances[i] = instr["distance"]
                break

    # distance gain as a smooth per-sample curve (same law as spatialise_stem)
    gains  = 1.0 / (1.0 + distances * 3.0)
    times  = np.arange(len(gains)) * resolution
    gain_t = np.interp(np.arange(n_samples) / sr, times, gains).astype(np.float32)

    return spatialise_trajectory(audio_mono * gain_t, azimuths, resolution,
                                 sofa_data, sr)


def render_binaural(stems_dir, job_id, spatial_result, sr=44100,
                    output_path=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, job_id + "_spatial.wav")

    print("[HRTF] Starting binaural rendering | Job: " + job_id)
    sofa_data = load_sofa()

    stem_files = {
        "vocals":       os.path.join(stems_dir, job_id + "_vocals.wav"),
        "instrumental": os.path.join(stems_dir, job_id + "_instrumental.wav"),
    }

    # Use fixed stem label map from renderer if available,
    # otherwise fall back to timeline-based detection
    stem_label_map = spatial_result.get("stem_labels", {})

    timeline  = spatial_result["timeline"]
    last_t    = timeline[-1]["time_sec"]
    n_samples = int((last_t + 1.0) * sr)
    mix_l     = np.zeros(n_samples, dtype=np.float32)
    mix_r     = np.zeros(n_samples, dtype=np.float32)

    for stem_name, stem_path in stem_files.items():
        if not os.path.exists(stem_path):
            print("[HRTF] Stem not found, skipping: " + stem_path)
            continue

        print("[HRTF] Processing stem: " + stem_name)
        audio, file_sr = sf.read(stem_path)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))

        # Use fixed label map first, then fall back to timeline detection
        stem_label = stem_label_map.get(stem_name, None)
        if stem_label is None:
            stem_label = _find_stem_label_from_timeline(stem_name, timeline)

        print("[HRTF]   Stem label: " + str(stem_label))

        binaural = spatialise_stem_with_motion(
            audio, timeline, stem_label, sofa_data, sr
        )
        mix_l += binaural[0]
        mix_r += binaural[1]

    peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-8)
    if peak > 0.95:
        mix_l = mix_l / peak * 0.95
        mix_r = mix_r / peak * 0.95

    output_stereo = np.stack([mix_l, mix_r], axis=1)
    sf.write(output_path, output_stereo, sr)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("[HRTF] Saved: " + output_path
          + " (" + str(round(size_mb, 1)) + " MB)")
    return output_path


def _find_stem_label_from_timeline(stem_name, timeline):
    """Fallback: find most common instrument label in timeline."""
    label_counts = {}
    for entry in timeline:
        for instr in entry.get("instruments", []):
            label = instr.get("raw_label", instr.get("label", ""))
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
    if not label_counts:
        return None
    return max(label_counts, key=label_counts.get)


def render_binaural_simple(stems_dir, job_id, detection_summary,
                            sr=44100, output_path=None):
    """Simplified rendering with fixed positions — for Phase 8 testing."""
    from core.spatial import INSTRUMENT_POSITIONS, DEFAULT_POSITION

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, job_id + "_spatial_simple.wav")

    print("[HRTF] Simple binaural rendering (fixed positions)")
    sofa_data = load_sofa()

    stem_files = {
        "vocals":       os.path.join(stems_dir, job_id + "_vocals.wav"),
        "instrumental": os.path.join(stems_dir, job_id + "_instrumental.wav"),
    }

    # Fixed positions: vocals center, instrumental front-left
    stem_positions = {
        "vocals":       {"azimuth": 0.0,   "distance": 0.3},
        "instrumental": {"azimuth": 315.0, "distance": 0.35},
    }

    n_samples = None
    mix_l     = None
    mix_r     = None

    for stem_name, stem_path in stem_files.items():
        if not os.path.exists(stem_path):
            print("[HRTF] Skipping missing stem: " + stem_path)
            continue

        audio, file_sr = sf.read(stem_path)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        if n_samples is None:
            n_samples = len(audio)
            mix_l     = np.zeros(n_samples, dtype=np.float32)
            mix_r     = np.zeros(n_samples, dtype=np.float32)

        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))

        pos      = stem_positions[stem_name]
        azimuth  = pos["azimuth"]
        distance = pos["distance"]

        print("[HRTF] Stem: " + stem_name
              + " -> az=" + str(azimuth) + "deg, dist=" + str(distance))

        binaural = spatialise_stem(audio, azimuth, distance, sofa_data, sr)
        mix_l   += binaural[0]
        mix_r   += binaural[1]

    if mix_l is None:
        print("[HRTF] No stems found")
        return None

    peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-8)
    if peak > 0.95:
        mix_l = mix_l / peak * 0.95
        mix_r = mix_r / peak * 0.95

    output_stereo = np.stack([mix_l, mix_r], axis=1)
    sf.write(output_path, output_stereo, sr)
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print("[HRTF] Saved: " + output_path + " (" + str(round(size_mb,1)) + " MB)")
    return output_path
