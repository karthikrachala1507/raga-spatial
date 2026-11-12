# Raga Spatial - core/separator8.py
# 8-stem separation with the custom BSRNN model (train_separator.py).
#
# The model works at 16 kHz mono, which on its own would cap every stem at
# 8 kHz. To keep the full 44.1 kHz bandwidth, the model's output is used only
# to decide HOW MUCH of each time-frequency cell belongs to each stem. Those
# shares (soft masks that sum to 1) are then applied to the original 44.1 kHz
# audio. Above 7.5 kHz, where the model cannot see, each stem gets the share
# it holds in the 4-7.5 kHz band at that moment. The stems therefore always
# add back up to the input.
#
# Two modes:
#   hybrid (default): vocals come from MelBandRoformer (much stronger on
#                     voice); the custom model splits the instrumental into
#                     the other 7 stems.
#   pure:             all 8 stems come from the custom model.

import os
import sys
import numpy as np
import torch
import soundfile as sf
import soxr

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEMS_DIR   = os.path.join(BACKEND_DIR, "outputs", "stems")

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]

# Where to look for the trained model, in order. RAGA_SEP8_MODEL overrides.
MODEL_CANDIDATES = [
    os.path.join(BACKEND_DIR, "models", "separator_v2_best.pt"),
    r"D:\raga-spatial-data\models\separator_v2_best.pt",
]

MODEL_SR    = 16000
CHUNK       = MODEL_SR * 10        # the model was trained on 10 s chunks
CHUNK_HOP   = MODEL_SR * 8         # 2 s overlap between chunks
FB_NFFT     = 4096                 # full-band STFT (93 ms window at 44.1 kHz)
FB_HOP      = 1024
SPLIT_HZ    = 7500.0               # masks from the model below, band shares above
HF_REF_HZ   = 4000.0               # lower edge of the band used for those shares
MIN_SHARE   = 0.005                # stems holding less than 0.5% are treated as absent

_MODEL_CACHE = {}


def find_model():
    override = os.environ.get("RAGA_SEP8_MODEL")
    for path in ([override] if override else []) + MODEL_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return None


def is_available():
    return find_model() is not None


def pick_device():
    """GPU when it has room (training may be using it), otherwise CPU."""
    forced = os.environ.get("RAGA_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free > 3.0e9:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def load_model(device):
    path = find_model()
    if path is None:
        raise FileNotFoundError("No 8-stem model found (separator_v2_best.pt)")
    key = (path, os.path.getmtime(path), device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)
    from train_separator import BandSplitRNN
    ckpt  = torch.load(path, map_location="cpu")
    model = BandSplitRNN(n_stems=len(CATEGORIES))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    epoch = int(ckpt.get("epoch", -1)) + 1
    print("[Separator8] Loaded " + path + " (epoch " + str(epoch) + ") on " + device.upper())
    _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = (model, epoch)
    return model, epoch


def infer_16k(model, audio_16k, device, progress=None):
    """Run the model over a whole song in overlapping 10 s chunks. Returns (8, n)."""
    n      = len(audio_16k)
    out    = np.zeros((len(CATEGORIES), n), dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    fade   = CHUNK - CHUNK_HOP
    ramp   = np.linspace(0.0, 1.0, fade, dtype=np.float32)

    starts = list(range(0, max(n - fade, 1), CHUNK_HOP))
    for i, start in enumerate(starts):
        seg = audio_16k[start:start + CHUNK]
        seg_len = len(seg)
        if seg_len < CHUNK:
            seg = np.pad(seg, (0, CHUNK - seg_len))
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))[None].to(device)
            y = model(x)[0].float().cpu().numpy()
        w = np.ones(CHUNK, dtype=np.float32)
        if i > 0:
            w[:fade] = ramp
        if i < len(starts) - 1:
            w[-fade:] = ramp[::-1]
        out[:, start:start + seg_len] += y[:, :seg_len] * w[:seg_len]
        weight[start:start + seg_len] += w[:seg_len]
        if progress and (i % 5 == 0 or i == len(starts) - 1):
            progress("8-stem model: chunk " + str(i + 1) + "/" + str(len(starts)))
    return out / np.maximum(weight, 1e-8)


def _stft(x, window):
    return torch.stft(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                      FB_NFFT, FB_HOP, window=window, return_complex=True)


def fullband_stems(mix, estimates_16k, sr):
    """
    Apply the model's estimates as soft masks to the full-band signal.

    mix           : (n,) mono signal at sr - the signal to split
    estimates_16k : (k, n16) model outputs at 16 kHz for the k stems to split into
    Returns (k, n) stems that sum to mix, and each stem's share of the energy.
    """
    n = len(mix)
    k = estimates_16k.shape[0]
    window  = torch.hann_window(FB_NFFT)
    mix_tf  = _stft(mix, window)
    n_bins  = mix_tf.shape[0]
    split   = min(int(SPLIT_HZ / (sr / FB_NFFT)), n_bins)
    hf_from = int(HF_REF_HZ / (sr / FB_NFFT))

    # power of each estimate below the split, on the same grid as the mix
    power = torch.zeros((k, split, mix_tf.shape[1]), dtype=torch.float32)
    for i in range(k):
        up = soxr.resample(estimates_16k[i], MODEL_SR, sr).astype(np.float32)
        up = np.pad(up, (0, max(0, n - len(up))))[:n]
        power[i] = _stft(up, window)[:split].abs() ** 2

    energy = power.sum(dim=(1, 2))
    shares = (energy / (energy.sum() + 1e-12)).numpy()
    for i in range(k):
        if shares[i] < MIN_SHARE:
            power[i].zero_()                 # absent stem: give its crumbs to the others
    shares = (power.sum(dim=(1, 2)) / (power.sum() + 1e-12)).numpy()

    eps   = 1e-10 * float(power.max()) + 1e-20
    total = power.sum(dim=0) + eps

    # share of the upper band per frame, smoothed over ~0.2 s
    hf = power[:, hf_from:split].sum(dim=1)                       # (k, frames)
    hf = torch.nn.functional.avg_pool1d(hf[None], 9, stride=1, padding=4)[0]
    hf_share = hf / (hf.sum(dim=0, keepdim=True) + eps)

    stems = np.zeros((k, n), dtype=np.float32)
    mask  = torch.empty((n_bins, mix_tf.shape[1]), dtype=torch.float32)
    for i in range(k):
        if shares[i] <= 0.0:
            continue
        mask[:split] = power[i] / total
        mask[split:] = hf_share[i][None, :]
        stems[i] = torch.istft(mix_tf * mask, FB_NFFT, FB_HOP, window=window,
                               length=n).numpy()
    return stems, shares


def _load_mono(path, sr, n):
    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = soxr.resample(audio, file_sr, sr).astype(np.float32)
    return np.pad(audio, (0, max(0, n - len(audio))))[:n]


def separate_8stems(audio, sr, job_id, vocals_path=None, instrumental_path=None,
                    progress=None):
    """
    Split a song into the 8 category stems at full bandwidth.

    audio : mono signal at sr (44.1 kHz in the pipeline)
    vocals_path / instrumental_path : MelBandRoformer outputs -> hybrid mode.
    Returns {"mode", "model_epoch", "device", "paths": {stem: wav}, "audio": {stem: array}}
    """
    os.makedirs(STEMS_DIR, exist_ok=True)
    device       = pick_device()
    model, epoch = load_model(device)
    n            = len(audio)
    hybrid       = bool(vocals_path and instrumental_path
                        and os.path.exists(vocals_path)
                        and os.path.exists(instrumental_path))

    if hybrid:
        vocals = _load_mono(vocals_path, sr, n)
        source = _load_mono(instrumental_path, sr, n)
        names  = [c for c in CATEGORIES if c != "vocals"]
    else:
        vocals = None
        source = np.asarray(audio, dtype=np.float32)
        names  = list(CATEGORIES)

    print("[Separator8] Mode: " + ("hybrid" if hybrid else "pure")
          + " | " + str(round(n / sr, 1)) + "s | device " + device.upper())
    source_16k = soxr.resample(source, sr, MODEL_SR).astype(np.float32)
    estimates  = infer_16k(model, source_16k, device, progress)
    keep       = [CATEGORIES.index(c) for c in names]
    stems, shares = fullband_stems(source, estimates[keep], sr)

    audio_out = {name: stems[i] for i, name in enumerate(names)}
    if hybrid:
        audio_out["vocals"] = vocals

    paths = {}
    for name in CATEGORIES:
        path = os.path.join(STEMS_DIR, job_id + "_" + name + ".wav")
        sf.write(path, np.clip(audio_out[name], -1.0, 1.0), sr, subtype="PCM_16")
        paths[name] = path
    print("[Separator8] Instrument shares: "
          + ", ".join(names[i] + " " + str(round(float(shares[i]) * 100)) + "%"
                      for i in range(len(names))))

    return {
        "mode":        "hybrid" if hybrid else "pure",
        "model_epoch": epoch,
        "device":      device,
        "paths":       paths,
        "audio":       audio_out,
    }
