# Raga Spatial - core/separator_2stem.py
# BS-Roformer (viperx, ep 317, SDR 12.97) source separation: vocals + instrumental.
# Used alone when no 8-stem model exists, and for the vocals in hybrid mode.
#
# Until 25 Nov 2026 this loaded MelBandRoformer.ckpt into a newer version of the
# bs_roformer package with strict=False; 842 weights stayed random, so the vocals
# were wrong. The checkpoint predates the package's hyper connections, so they are
# switched off here and the weights are loaded strictly.

import re
import torch
import numpy as np
import soundfile as sf
import librosa
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
STEMS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "stems")

VOCAL_MODEL = os.path.join(MODELS_DIR, "model_bs_roformer_ep_317_sdr_12.9755.ckpt")

TARGET_SR  = 44100
CHUNK_SIZE = 352800
OVERLAP    = 2          # 4 took twice as long for the same quality (25 Nov 2026)
MIN_CHUNK  = 8192
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def is_usable():
    """True when the checkpoint is there and the GPU has room (about 4 GB)."""
    if not os.path.exists(VOCAL_MODEL) or not torch.cuda.is_available():
        return False
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:
        return False
    return free > 4.0e9


def separate_stems(audio, sr, job_id):
    print("[Separator] Starting separation on " + DEVICE.upper())
    print("[Separator] Audio duration: " + str(round(len(audio) / sr, 2)) + "s")
    os.makedirs(STEMS_DIR, exist_ok=True)
    model = load_model()
    audio_tensor = prepare_audio_tensor(audio, sr)
    print("[Separator] Audio tensor shape: " + str(audio_tensor.shape))
    print("[Separator] Separating stems...")
    vocals_audio, instrumental_audio = run_separation(model, audio_tensor, sr)
    del model                      # the 8-stem model and BEATs need the GPU memory next
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    vocals_path       = os.path.join(STEMS_DIR, job_id + "_vocals.wav")
    instrumental_path = os.path.join(STEMS_DIR, job_id + "_instrumental.wav")
    save_stem(vocals_audio, sr, vocals_path)
    save_stem(instrumental_audio, sr, instrumental_path)
    print("[Separator] Separation complete")
    return {"vocals": vocals_path, "instrumental": instrumental_path}


def load_model():
    from bs_roformer import BSRoformer

    if not os.path.exists(VOCAL_MODEL):
        raise FileNotFoundError(os.path.basename(VOCAL_MODEL) + " not found in " + MODELS_DIR)
    print("[Separator] Loading BS-Roformer from: " + VOCAL_MODEL)

    model = BSRoformer(
        dim=512,
        depth=12,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head=64,
        heads=8,
        mask_estimator_depth=2,
        stft_hop_length=441,
        num_residual_streams=1,          # plain residuals, as the checkpoint was trained
    )
    # the checkpoint has no "value residual" layers either
    for module in model.modules():
        if hasattr(module, "to_value_residual_mix"):
            module.to_value_residual_mix = None

    checkpoint = torch.load(VOCAL_MODEL, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    # the package now wraps each attention / feed-forward block in ".branch"
    state_dict = {re.sub(r"^(layers\.\d+\.\d+\.layers\.\d+\.\d+)\.", r"\g<1>.branch.", k): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)    # strict: a wrong checkpoint must fail loudly
    print("[Separator] Loaded all " + str(len(state_dict)) + " weights")
    return model.to(DEVICE).eval()


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


def infer(model, x):
    if DEVICE == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            return model(x).float()
    return model(x)


def run_separation(model, audio_tensor, sr):
    with torch.no_grad():
        total_samples = audio_tensor.shape[-1]
        if total_samples <= CHUNK_SIZE * 2:
            print("[Separator] Short audio - single pass inference")
            vocals_tensor = infer(model, audio_tensor)
            orig_len      = total_samples
            vocals        = vocals_tensor.squeeze(0).cpu().numpy()
            if vocals.ndim == 3:
                vocals    = vocals[0]
            vocals        = vocals[:, :orig_len]
            # The model returns slightly fewer samples than it was given.
            # Pad the tail with silence so songs under 16 s do not crash.
            if vocals.shape[-1] < orig_len:
                vocals    = np.pad(vocals, ((0, 0), (0, orig_len - vocals.shape[-1])))
            original      = audio_tensor.squeeze(0).cpu().numpy()
            instrumental  = original - vocals
            return vocals, instrumental
        else:
            print("[Separator] Long audio - chunked inference")
            vocals       = process_in_chunks(model, audio_tensor)
            original     = audio_tensor.squeeze(0).cpu().numpy()
            instrumental = original - vocals
            return vocals, instrumental


def process_in_chunks(model, audio_tensor):
    audio         = audio_tensor.squeeze(0)
    total_samples = audio.shape[-1]
    hop           = CHUNK_SIZE // OVERLAP

    output = np.zeros((2, total_samples), dtype=np.float32)
    count  = np.zeros(total_samples,      dtype=np.float32)

    # Hann window for smooth overlap-add (prevents clicking at chunk boundaries)
    window = np.hanning(CHUNK_SIZE).astype(np.float32)

    chunk_num = 0
    start     = 0

    while start < total_samples:
        end            = min(start + CHUNK_SIZE, total_samples)
        chunk          = audio[:, start:end]
        orig_chunk_len = chunk.shape[-1]

        if orig_chunk_len < MIN_CHUNK:
            print("[Separator] Skipping tiny tail chunk of " + str(orig_chunk_len) + " samples")
            break

        if orig_chunk_len < CHUNK_SIZE:
            pad_size = CHUNK_SIZE - orig_chunk_len
            chunk    = torch.nn.functional.pad(chunk, (0, pad_size))

        chunk_in = chunk.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = infer(model, chunk_in)

        out_np = out.squeeze(0).cpu().numpy()
        if out_np.ndim == 3:
            out_np = out_np[0]

        # Use actual model output length - never assume it equals input length
        actual_out_len = out_np.shape[-1]
        usable_len     = min(actual_out_len, orig_chunk_len)
        end_idx        = min(start + usable_len, total_samples)
        copy_len       = end_idx - start
        out_np         = out_np[:, :copy_len]

        win_slice = window[:copy_len]
        output[:, start:end_idx] += out_np * win_slice[np.newaxis, :]
        count[start:end_idx]     += win_slice

        chunk_num += 1
        print("[Separator] Chunk " + str(chunk_num) + ": "
              + str(round(start / TARGET_SR, 1)) + "s to "
              + str(round(end_idx / TARGET_SR, 1)) + "s"
              + "  (model returned " + str(actual_out_len) + " samples)")

        start += hop

    count  = np.maximum(count, 1e-8)
    output = output / count[np.newaxis, :]

    return output


def save_stem(audio, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 2:
        audio = audio.T
    sf.write(path, audio, sr, subtype="FLOAT")   # float keeps vocals + instrumental = input
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print("[Separator] Saved: " + path + " (" + str(round(size_mb, 1)) + " MB)")
