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

TARGET_SR  = 44100
CHUNK_SIZE = 352800
OVERLAP    = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def separate_stems(audio, sr, job_id):
    print("[Separator] Starting separation on " + DEVICE.upper())
    print("[Separator] Audio duration: " + str(round(len(audio)/sr, 2)) + "s")
    os.makedirs(STEMS_DIR, exist_ok=True)
    model = load_model()
    audio_tensor = prepare_audio_tensor(audio, sr)
    print("[Separator] Audio tensor shape: " + str(audio_tensor.shape))
    print("[Separator] Separating stems...")
    vocals_audio, instrumental_audio = run_separation(model, audio_tensor, sr)
    vocals_path = os.path.join(STEMS_DIR, job_id + "_vocals.wav")
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

    # Exact architecture from checkpoint inspection:
    # dim=384, depth=6, num_bands=60
    # mask_estimators.0.to_freqs.0.0.0 -> 3 layers = mask_estimator_depth=2
    # time/freq transformer depth=2
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

    result = model.load_state_dict(new_state_dict, strict=False)
    missing = len(result.missing_keys)
    unexpected = len(result.unexpected_keys)
    print("[Separator] Loaded. Missing: " + str(missing) + " Unexpected: " + str(unexpected))

    if missing == 0 and unexpected == 0:
        print("[Separator] Perfect match - model loaded successfully!")
    elif missing < 10:
        print("[Separator] Minor mismatch but usable - continuing...")
    else:
        print("[WARN] " + str(missing) + " missing keys")

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
        if audio_tensor.shape[-1] <= CHUNK_SIZE * 2:
            print("[Separator] Short audio - single pass inference")
            vocals_tensor = model(audio_tensor)
            vocals = vocals_tensor.squeeze(0).cpu().numpy()
            original = audio_tensor.squeeze(0).cpu().numpy()
            instrumental = original - vocals
            return vocals, instrumental
        else:
            print("[Separator] Long audio - chunked inference")
            vocals = process_in_chunks(model, audio_tensor)
            original = audio_tensor.squeeze(0).cpu().numpy()
            instrumental = original - vocals
            return vocals, instrumental


def process_in_chunks(model, audio_tensor):
    audio = audio_tensor.squeeze(0)
    total_samples = audio.shape[-1]
    hop = CHUNK_SIZE // OVERLAP
    output = np.zeros((2, total_samples), dtype=np.float32)
    count  = np.zeros(total_samples, dtype=np.float32)
    chunk_num = 0
    start = 0
    while start < total_samples:
        end = min(start + CHUNK_SIZE, total_samples)
        chunk = audio[:, start:end]
        if chunk.shape[-1] < CHUNK_SIZE:
            pad_size = CHUNK_SIZE - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        chunk = chunk.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(chunk)
        out_np = out.squeeze(0).cpu().numpy()
        actual_len = min(CHUNK_SIZE, total_samples - start)
        output[:, start:start+actual_len] += out_np[:, :actual_len]
        count[start:start+actual_len] += 1.0
        chunk_num += 1
        print("[Separator] Chunk " + str(chunk_num) + ": " + str(round(start/44100,1)) + "s to " + str(round(end/44100,1)) + "s")
        start += hop
    count = np.maximum(count, 1.0)
    output = output / count[np.newaxis, :]
    return output


def save_stem(audio, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 2:
        audio = audio.T
    sf.write(path, audio, sr)
    size_mb = os.path.getsize(path) / (1024*1024)
    print("[Separator] Saved: " + path + " (" + str(round(size_mb,1)) + " MB)")
