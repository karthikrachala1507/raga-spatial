# Raga Spatial - core/separator.py
# MelBandRoformer Source Separation Module

import torch
import numpy as np
import soundfile as sf
import librosa
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
STEMS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "stems")

MELBAND_MODEL    = os.path.join(MODELS_DIR, "MelBandRoformer.ckpt")
BSROFORMER_MODEL = os.path.join(MODELS_DIR, "model_bs_roformer_ep_317_sdr_12.9755.ckpt")

TARGET_SR     = 44100
CHUNK_SIZE    = 352800   # ~8 seconds at 44100Hz
OVERLAP       = 4        # 75% overlap (hop = CHUNK_SIZE // OVERLAP)
MIN_CHUNK     = 8192     # minimum samples to feed the model safely
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def separate_stems(audio, sr, job_id):
    print("[Separator] Starting separation on " + DEVICE.upper())
    print("[Separator] Audio duration: " + str(round(len(audio) / sr, 2)) + "s")
    os.makedirs(STEMS_DIR, exist_ok=True)
    model = load_model()
    audio_tensor = prepare_audio_tensor(audio, sr)
    print("[Separator] Audio tensor shape: " + str(audio_tensor.shape))
    print("[Separator] Separating stems...")
    vocals_audio, instrumental_audio = run_separation(model, audio_tensor, sr)
    vocals_path       = os.path.join(STEMS_DIR, job_id + "_vocals.wav")
    instrumental_path = os.path.join(STEMS_DIR, job_id + "_instrumental.wav")
    save_stem(vocals_audio, sr, vocals_path)
    save_stem(instrumental_audio, sr, instrumental_path)
    print("[Separator] Separation complete")
    return {"vocals": vocals_path, "instrumental": instrumental_path}


def load_model():
    from bs_roformer import MelBandRoformer

    model_path = None
    if os.path.exists(MELBAND_MODEL):
        model_path = MELBAND_MODEL
        print("[Separator] Loading MelBandRoformer from: " + MELBAND_MODEL)
    elif os.path.exists(BSROFORMER_MODEL):
        model_path = BSROFORMER_MODEL
        print("[Separator] Loading from: " + BSROFORMER_MODEL)
    else:
        raise FileNotFoundError("No model found in " + MODELS_DIR)

    model = MelBandRoformer(
        dim=384,
        depth=6,
        stereo=True,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        num_bands=60,
        dim_head=64,
        heads=8,
        mask_estimator_depth=2,
    )

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")
        new_state_dict[new_key] = v

    result  = model.load_state_dict(new_state_dict, strict=False)
    missing = len(result.missing_keys)
    unexpected = len(result.unexpected_keys)
    print("[Separator] Loaded. Missing: " + str(missing) + " Unexpected: " + str(unexpected))
    model = model.to(DEVICE)
    model.eval()
    return model


def prepare_audio_tensor(audio, sr):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if audio.ndim == 1:
        audio_stereo = np.stack([audio, audio], axis=0)
    else:
        audio_stereo = audio
    tensor = torch.tensor(audio_stereo, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    return tensor.to(DEVICE)


def run_separation(model, audio_tensor, sr):
    with torch.no_grad():
        total_samples = audio_tensor.shape[-1]

        if total_samples <= CHUNK_SIZE * 2:
            print("[Separator] Short audio - single pass inference")
            vocals_tensor = model(audio_tensor)
            # model output may differ in length due to STFT padding - always trim
            orig_len  = total_samples
            vocals    = vocals_tensor.squeeze(0).cpu().numpy()
            vocals    = vocals[:, :orig_len]
            original  = audio_tensor.squeeze(0).cpu().numpy()
            instrumental = original - vocals
            return vocals, instrumental
        else:
            print("[Separator] Long audio - chunked inference")
            vocals = process_in_chunks(model, audio_tensor)
            original = audio_tensor.squeeze(0).cpu().numpy()
            instrumental = original - vocals
            return vocals, instrumental


def process_in_chunks(model, audio_tensor):
    audio         = audio_tensor.squeeze(0)           # shape: (2, total_samples)
    total_samples = audio.shape[-1]
    hop           = CHUNK_SIZE // OVERLAP              # 88200 samples = ~2s hop

    output = np.zeros((2, total_samples), dtype=np.float32)
    count  = np.zeros(total_samples,      dtype=np.float32)

    # Hann window for smooth overlap-add (avoids clicks at chunk boundaries)
    window = np.hanning(CHUNK_SIZE).astype(np.float32)  # shape: (CHUNK_SIZE,)

    chunk_num = 0
    start     = 0

    while start < total_samples:
        end            = min(start + CHUNK_SIZE, total_samples)
        chunk          = audio[:, start:end]           # (2, chunk_len)
        orig_chunk_len = chunk.shape[-1]

        # Skip tiny tail chunks the model cannot process reliably
        if orig_chunk_len < MIN_CHUNK:
            print("[Separator] Skipping tiny tail chunk of " + str(orig_chunk_len) + " samples")
            break

        # Pad short chunk to CHUNK_SIZE so model always gets a full-size input
        if orig_chunk_len < CHUNK_SIZE:
            pad_size = CHUNK_SIZE - orig_chunk_len
            chunk    = torch.nn.functional.pad(chunk, (0, pad_size))

        chunk_in = chunk.unsqueeze(0).to(DEVICE)       # (1, 2, CHUNK_SIZE)

        with torch.no_grad():
            out = model(chunk_in)                      # (1, 1, 2, out_len) or (1, 2, out_len)

        # Normalise output shape to (2, out_len)
        out_np = out.squeeze(0).cpu().numpy()
        if out_np.ndim == 3:
            out_np = out_np[0]                         # drop stem dimension if present

        # --- THE FIX ---
        # Use actual model output length, NOT assumed CHUNK_SIZE or orig_chunk_len.
        # MelBandRoformer STFT pads to FFT boundaries, so output is always slightly
        # shorter than input. Using the assumed length causes the broadcast crash.
        actual_out_len = out_np.shape[-1]

        # Trim to original (unpadded) chunk region and clip to total_samples
        usable_len = min(actual_out_len, orig_chunk_len)
        end_idx    = min(start + usable_len, total_samples)
        copy_len   = end_idx - start

        out_np = out_np[:, :copy_len]                  # (2, copy_len)

        # Apply Hann window weights for the copied region
        win_slice = window[:copy_len]                  # (copy_len,)

        output[:, start:end_idx] += out_np * win_slice[np.newaxis, :]
        count[start:end_idx]     += win_slice

        chunk_num += 1
        print("[Separator] Chunk " + str(chunk_num) + ": "
              + str(round(start / TARGET_SR, 1)) + "s to "
              + str(round(end_idx / TARGET_SR, 1)) + "s"
              + "  (model returned " + str(actual_out_len) + " samples)")

        start += hop

    # Normalise by accumulated window weights; avoid div-by-zero at uncovered tail
    count  = np.maximum(count, 1e-8)
    output = output / count[np.newaxis, :]
    return output


def save_stem(audio, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio    = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 2:
        audio = audio.T                                # soundfile expects (samples, channels)
    sf.write(path, audio, sr)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print("[Separator] Saved: " + path + " (" + str(round(size_mb, 1)) + " MB)")
