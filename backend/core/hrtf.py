# Raga Spatial - core/hrtf.py
# Phase 8: HRTF Binaural Rendering
#
# Loads MIT KEMAR SOFA file, extracts HRIRs for any azimuth/elevation,
# and convolves each instrument stem with the correct HRIR to produce
# binaural (left + right ear) output.
#
# Pipeline:
#   separated stems (wav) + spatial timeline (json)
#   -> per-stem HRTF convolution
#   -> mix all stems into final binaural output
#   -> output_spatial.wav

import numpy as np
import os
import soundfile as sf
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

MODELS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SOFA_FILE    = os.path.join(MODELS_DIR, "mit_kemar_normal_pinna.sofa")
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "spatial")

TARGET_SR    = 44100


# ── SOFA loader ───────────────────────────────────────────────────────────────

def load_sofa(sofa_path=None):
    """
    Load MIT KEMAR SOFA file and extract HRIR data.

    Returns dict:
    {
        "azimuths"   : np.ndarray (N,)   - azimuth angles in degrees
        "elevations" : np.ndarray (N,)   - elevation angles in degrees
        "hrir_left"  : np.ndarray (N, L) - left ear impulse responses
        "hrir_right" : np.ndarray (N, L) - right ear impulse responses
        "sr"         : int               - sample rate of HRIRs
        "n_taps"     : int               - length of each HRIR
    }
    """
    if sofa_path is None:
        sofa_path = SOFA_FILE

    if not os.path.exists(sofa_path):
        raise FileNotFoundError("SOFA file not found: " + sofa_path)

    print("[HRTF] Loading SOFA file: " + sofa_path)

    try:
        import pysofaconventions as pysofa
        sofa      = pysofa.SOFAFile(sofa_path, "r")
        positions = sofa.getVariableValue("SourcePosition")  # (N, 3): az, el, dist
        data      = sofa.getVariableValue("Data.IR")         # (N, 2, L)
        sr        = int(sofa.getVariableValue("Data.SamplingRate").item())
        sofa.close()

        azimuths   = positions[:, 0].astype(np.float32)
        elevations = positions[:, 1].astype(np.float32)
        hrir_left  = data[:, 0, :].astype(np.float32)   # (N, L)
        hrir_right = data[:, 1, :].astype(np.float32)   # (N, L)

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
    """
    Find the closest HRIR measurement for a given azimuth and elevation.
    SOFA uses interaural azimuth (0=front, 90=left in MIT KEMAR).
    We use our convention (0=front, 90=right, 270=left) so convert.

    Returns (hrir_left, hrir_right) each shape (L,)
    """
    # Convert our azimuth convention to MIT KEMAR convention
    # Our:     0=front, 90=right, 180=rear, 270=left
    # MIT:     0=front, 90=left,  180=rear, 270=right
    # So:  MIT_az = (360 - our_az) % 360
    mit_az = (360.0 - azimuth_deg) % 360.0

    azimuths   = sofa_data["azimuths"]
    elevations = sofa_data["elevations"]

    # Find measurements at closest elevation
    elev_diffs  = np.abs(elevations - elevation_deg)
    min_elev    = np.min(elev_diffs)
    elev_mask   = elev_diffs <= min_elev + 1.0   # allow 1 degree tolerance

    # Among those, find closest azimuth
    filtered_az  = azimuths[elev_mask]
    filtered_idx = np.where(elev_mask)[0]

    az_diffs = np.abs(filtered_az - mit_az)
    # Handle wraparound (e.g. distance between 350 and 10 degrees)
    az_diffs = np.minimum(az_diffs, 360.0 - az_diffs)

    closest = filtered_idx[np.argmin(az_diffs)]

    return (sofa_data["hrir_left"][closest],
            sofa_data["hrir_right"][closest])


# ── Per-stem HRTF convolution ─────────────────────────────────────────────────

def spatialise_stem(audio_mono, azimuth_deg, distance, sofa_data,
                    stem_sr=44100):
    """
    Apply HRTF convolution to a mono audio stem.

    Args:
        audio_mono  : np.ndarray (N,) mono audio
        azimuth_deg : azimuth in degrees (our convention)
        distance    : 0.0-1.0 (affects gain only)
        sofa_data   : loaded SOFA data from load_sofa()
        stem_sr     : sample rate of audio_mono

    Returns:
        np.ndarray (2, N) stereo binaural audio (left, right)
    """
    hrir_l, hrir_r = get_hrir_for_azimuth(sofa_data, azimuth_deg)

    # Resample HRIRs if SOFA SR != stem SR
    if sofa_data["sr"] != stem_sr:
        from scipy.signal import resample
        target_len = int(len(hrir_l) * stem_sr / sofa_data["sr"])
        hrir_l = resample(hrir_l, target_len)
        hrir_r = resample(hrir_r, target_len)

    # Convolve mono signal with left and right HRIRs
    left  = fftconvolve(audio_mono, hrir_l, mode="full")
    right = fftconvolve(audio_mono, hrir_r, mode="full")

    # Trim to original length
    left  = left[:len(audio_mono)]
    right = right[:len(audio_mono)]

    # Apply distance-based gain (inverse square law approximation)
    # distance 0.0 = full gain, 1.0 = -12dB
    gain  = 1.0 / (1.0 + distance * 3.0)
    left  *= gain
    right *= gain

    return np.stack([left, right], axis=0)   # (2, N)


def spatialise_stem_with_motion(audio_mono, spatial_timeline,
                                 stem_label, sofa_data, sr=44100):
    """
    Spatialise a stem with time-varying position (supports motion events).

    Divides audio into chunks matching the spatial timeline resolution,
    applies the correct HRTF for each chunk, then crossfades between them.

    Args:
        audio_mono      : np.ndarray (N,) mono audio
        spatial_timeline: list of timeline entries from spatial.py
        stem_label      : instrument label to look up in timeline
        sofa_data       : loaded SOFA data
        sr              : sample rate

    Returns:
        np.ndarray (2, N) stereo binaural audio
    """
    n_samples   = len(audio_mono)
    output      = np.zeros((2, n_samples), dtype=np.float32)
    resolution  = spatial_timeline[1]["time_sec"] - spatial_timeline[0]["time_sec"]
    chunk_size  = int(resolution * sr)
    fade_size   = min(int(0.02 * sr), chunk_size // 4)  # 20ms crossfade

    for i, entry in enumerate(spatial_timeline):
        start_sample = int(entry["time_sec"] * sr)
        end_sample   = min(start_sample + chunk_size, n_samples)

        if start_sample >= n_samples:
            break

        # Find this stem's position at this time
        azimuth  = 0.0
        distance = 0.5
        for instr in entry.get("instruments", []):
            if instr.get("raw_label") == stem_label or instr.get("label") == stem_label:
                azimuth  = instr["azimuth"]
                distance = instr["distance"]
                break

        chunk = audio_mono[start_sample:end_sample]
        if len(chunk) == 0:
            continue

        # Spatialise this chunk
        spat = spatialise_stem(chunk, azimuth, distance, sofa_data, sr)

        # Apply crossfade at boundaries to avoid clicks
        if fade_size > 0 and len(chunk) > fade_size * 2:
            fade_in  = np.linspace(0.0, 1.0, fade_size)
            fade_out = np.linspace(1.0, 0.0, fade_size)
            spat[:, :fade_size]   *= fade_in
            spat[:, -fade_size:]  *= fade_out

        output[:, start_sample:end_sample] += spat

    return output


# ── Full rendering pipeline ───────────────────────────────────────────────────

def render_binaural(stems_dir, job_id, spatial_result, sr=44100,
                    output_path=None):
    """
    Full binaural rendering pipeline.

    Loads separated stems, spatialises each one according to the
    spatial timeline, mixes them together, and saves output_spatial.wav.

    Args:
        stems_dir     : directory containing stem wav files
        job_id        : job identifier (used in stem filenames)
        spatial_result: output of spatial.assign_spatial_positions()
        sr            : sample rate
        output_path   : output wav path (auto-generated if None)

    Returns:
        path to output_spatial.wav
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, job_id + "_spatial.wav")

    print("[HRTF] Starting binaural rendering")
    print("[HRTF] Job: " + job_id)

    # Load SOFA
    sofa_data = load_sofa()

    # Load available stems
    stem_files = {
        "vocals":       os.path.join(stems_dir, job_id + "_vocals.wav"),
        "instrumental": os.path.join(stems_dir, job_id + "_instrumental.wav"),
    }

    timeline = spatial_result["timeline"]

    # Determine total output length from timeline
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

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        # Resample if needed
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        # Trim or pad to expected length
        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))

        # Find the dominant instrument label for this stem in the timeline
        stem_label = _find_stem_label(stem_name, timeline)
        print("[HRTF]   Stem label: " + str(stem_label))

        # Spatialise with motion
        binaural = spatialise_stem_with_motion(
            audio, timeline, stem_label, sofa_data, sr
        )

        mix_l += binaural[0]
        mix_r += binaural[1]

    # Normalise to prevent clipping
    peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-8)
    if peak > 0.95:
        mix_l = mix_l / peak * 0.95
        mix_r = mix_r / peak * 0.95

    # Save stereo output
    output_stereo = np.stack([mix_l, mix_r], axis=1)  # (N, 2)
    sf.write(output_path, output_stereo, sr)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("[HRTF] Saved: " + output_path
          + " (" + str(round(size_mb, 1)) + " MB)")

    return output_path


def _find_stem_label(stem_name, timeline):
    """
    Find the most common instrument label associated with a stem
    across the timeline. Used to look up position in spatial data.
    """
    label_counts = {}
    for entry in timeline:
        for instr in entry.get("instruments", []):
            label = instr.get("raw_label", instr.get("label", ""))
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

    if not label_counts:
        return None

    return max(label_counts, key=label_counts.get)


# ── Simple static spatialiser (no timeline, just fixed positions) ─────────────

def render_binaural_simple(stems_dir, job_id, detection_summary,
                            sr=44100, output_path=None):
    """
    Simplified rendering without motion — each stem placed at a fixed position.
    Useful for testing Phase 8 independently before full pipeline is connected.

    Args:
        stems_dir         : directory with stem wavs
        job_id            : job id
        detection_summary : list of {"label", "avg_confidence"} from detector
        sr                : sample rate
        output_path       : output path (auto if None)

    Returns:
        path to output wav
    """
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

    # Build label -> position map from detections
    label_positions = {}
    for det in detection_summary:
        label = det["label"]
        pos   = INSTRUMENT_POSITIONS.get(label, None)
        if pos:
            label_positions[label] = pos

    n_samples = None
    mix_l     = None
    mix_r     = None

    stem_assignments = {
        "vocals":       0.0,    # front center
        "instrumental": 315.0,  # front left (plucked strings default)
    }

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

        # Trim/pad
        if len(audio) > n_samples:
            audio = audio[:n_samples]
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))

        azimuth  = stem_assignments.get(stem_name, 0.0)
        distance = 0.4

        # Use top detected instrument position if available
        if label_positions:
            top_label = list(label_positions.keys())[0]
            azimuth   = label_positions[top_label]["azimuth"]
            distance  = label_positions[top_label]["distance"]

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
