# Raga Spatial - evaluate_validation.py
# Measures how well a checkpoint SEPARATES, in dB, on the fixed validation
# mixes (built from source files the model never trained on).
#
# For every stem that is present in a mix it reports SDR in dB:
#   model   : the stem as the app produces it (model output turned into soft
#             masks on the mix, the same thing core/separator8.py does)
#   raw out : the model's direct output
#   mix     : what you get by using the whole mix as the stem (reference only)
#
# How to read "model":
#     0 dB  = no better than outputting silence. NOT separating.
#   1-3 dB  = starting to separate; stems will sound smeared with heavy bleed.
#   4-6 dB  = usable for spatial placement.
#    7+ dB  = good.
#
# Usage:
#   python evaluate_validation.py                       (best checkpoint, 80 mixes)
#   python evaluate_validation.py separator_v2_latest.pt 120
#   set RAGA_DEVICE=cpu   to keep the GPU free while training is running

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_separator as ts


def sdr(target, estimate):
    noise = target - estimate
    return 10 * np.log10((np.sum(target ** 2) + 1e-10) / (np.sum(noise ** 2) + 1e-10))


def wiener(mix, preds, window):
    """Model outputs -> soft masks that sum to 1 -> applied to the mix."""
    mix_tf = torch.stft(mix, 1024, 256, window=window, return_complex=True)
    power  = torch.stack([torch.stft(p, 1024, 256, window=window,
                                     return_complex=True).abs() ** 2 for p in preds])
    masks  = power / (power.sum(dim=0, keepdim=True) + 1e-10)
    return torch.stack([torch.istft(mix_tf * m, 1024, 256, window=window,
                                    length=mix.shape[-1]) for m in masks])


def main():
    name   = sys.argv[1] if len(sys.argv) > 1 else "separator_v2_best.pt"
    n_mix  = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    device = os.environ.get("RAGA_DEVICE", ts.DEVICE)
    path   = name if os.path.isabs(name) else os.path.join(ts.MODELS_DIR, name)

    ckpt  = torch.load(path, map_location="cpu")
    model = ts.BandSplitRNN(n_stems=ts.N_STEMS)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print("Checkpoint: " + path + " (epoch " + str(ckpt["epoch"] + 1) + ") on " + device)

    pool   = ts.ClipPool(ts.POOL_DIRS)
    loader = DataLoader(ts.ValMixDataset(pool, n_mix), batch_size=1)
    window = torch.hann_window(1024)
    rows   = {c: [] for c in ts.CATEGORIES}

    with torch.no_grad():
        for mix, targets, active in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                preds = model(mix.to(device))
            preds = preds.float().cpu()
            for b in range(mix.shape[0]):
                est = wiener(mix[b], preds[b], window).numpy()
                for s, cat in enumerate(ts.CATEGORIES):
                    if active[b, s] < 0.5:
                        continue
                    tgt = targets[b, s].numpy()
                    rows[cat].append((sdr(tgt, mix[b].numpy()), sdr(tgt, est[s]),
                                      sdr(tgt, preds[b, s].numpy())))

    print("\n%-16s %5s %9s %11s %8s" % ("stem", "n", "model dB", "raw out dB", "mix dB"))
    all_sdr = []
    for cat in ts.CATEGORIES:
        r = np.array(rows[cat])
        if len(r) == 0:
            continue
        all_sdr.extend(r[:, 1].tolist())
        print("%-16s %5d %9.1f %11.1f %8.1f" % (
            cat, len(r), np.median(r[:, 1]), np.median(r[:, 2]), np.median(r[:, 0])))
    med = float(np.median(all_sdr))
    verdict = ("not separating yet" if med < 1 else "starting to separate" if med < 4
               else "usable" if med < 7 else "good")
    print("\nOverall median: %.1f dB over %d stems -> %s (0 dB = silence)" % (
        med, len(all_sdr), verdict))


if __name__ == "__main__":
    main()
