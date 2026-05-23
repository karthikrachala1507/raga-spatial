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


def spatialise_stem_with_motion(audio_mono, spatial_timeline,
                                 stem_label, sofa_data, sr=44100):
    n_samples  = len(audio_mono)
    output     = np.zeros((2, n_samples), dtype=np.float32)
    resolution = spatial_timeline[1]["time_sec"] - spatial_timeline[0]["time_sec"]
    chunk_size = int(resolution * sr)
    fade_size  = min(int(0.02 * sr), chunk_size // 4)

    for i, entry in enumerate(spatial_timeline):
        start_sample = int(entry["time_sec"] * sr)
        end_sample   = min(start_sample + chunk_size, n_samples)
        if start_sample >= n_samples:
            break

        azimuth  = 0.0
        distance = 0.5
        for instr in entry.get("instruments", []):
            if (instr.get("raw_label") == stem_label or
                    instr.get("label") == stem_label):
                azimuth  = instr["azimuth"]
                distance = instr["distance"]
                break

        chunk = audio_mono[start_sample:end_sample]
        if len(chunk) == 0:
            continue

        spat = spatialise_stem(chunk, azimuth, distance, sofa_data, sr)

        if fade_size > 0 and len(chunk) > fade_size * 2:
            fade_in  = np.linspace(0.0, 1.0, fade_size)
            fade_out = np.linspace(1.0, 0.0, fade_size)
            spat[:, :fade_size]  *= fade_in
            spat[:, -fade_size:] *= fade_out

        output[:, start_sample:end_sample] += spat

    return output


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
