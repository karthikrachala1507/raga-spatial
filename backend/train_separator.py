# Raga Spatial - train_separator.py (FIXED v2)
# Trains 8-stem BSRNN separator on synthetic mixes
# Fixes silent-output collapse from v1:
#   1. Loss now ignores near-empty target stems instead of unstable SDR division
#   2. Added direct L1 + spectral magnitude loss as primary signal
#   3. Mask decoder no longer ends in raw Tanh alone - uses Tanh * learnable gain
#   4. Active-stem masking prevents 8-way gradient dilution from absent categories

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
from tqdm import tqdm

# v2 mixes: target stems are at their true level in the mix (stems sum to mix).
# Do NOT point this at the old synthetic_mixes folder - its targets are too loud.
MIXES_DIR  = r"D:\raga-spatial-data\synthetic_mixes_v2"
MODELS_DIR = r"D:\raga-spatial-data\models"
LOGS_DIR   = r"D:\raga-spatial-data\logs"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE  = 16000
DURATION     = 10.0
N_SAMPLES    = int(SAMPLE_RATE * DURATION)
N_STEMS      = 8
BATCH_SIZE   = 4
N_EPOCHS     = 100
LR           = 1e-3
LR_DECAY     = 0.98
SAVE_EVERY   = 10
VAL_SPLIT    = 0.1
SPLIT_SEED   = 1507     # fixed, so train/val split is identical on every resume
MIN_TARGET_RMS = 1e-4   # below this, a target stem is considered "absent"

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]

# ── Dataset ───────────────────────────────────────────────────────────────────

class StemDataset(Dataset):
    def __init__(self, mix_dirs):
        self.mix_dirs = mix_dirs

    def __len__(self):
        return len(self.mix_dirs)

    def __getitem__(self, idx):
        mix_dir = self.mix_dirs[idx]

        mix_path = os.path.join(mix_dir, "mix.wav")
        mix, _   = sf.read(mix_path)
        mix      = mix.astype(np.float32)
        if len(mix) < N_SAMPLES:
            mix = np.pad(mix, (0, N_SAMPLES - len(mix)))
        mix = mix[:N_SAMPLES]

        targets = []
        active  = []
        for cat in CATEGORIES:
            stem_path = os.path.join(mix_dir, cat + ".wav")
            if os.path.exists(stem_path):
                stem, _ = sf.read(stem_path)
                stem    = stem.astype(np.float32)
                if len(stem) < N_SAMPLES:
                    stem = np.pad(stem, (0, N_SAMPLES - len(stem)))
                stem = stem[:N_SAMPLES]
            else:
                stem = np.zeros(N_SAMPLES, dtype=np.float32)

            rms = float(np.sqrt(np.mean(stem ** 2)))
            active.append(1.0 if rms > MIN_TARGET_RMS else 0.0)
            targets.append(stem)

        targets = np.stack(targets, axis=0)
        active  = np.array(active, dtype=np.float32)

        return (torch.tensor(mix),
                torch.tensor(targets),
                torch.tensor(active))


# ── Model: BSRNN with gain-corrected mask decoder ─────────────────────────────

class BandSplitRNN(nn.Module):
    def __init__(self, n_stems=8, n_fft=1024, hop=256,
                 n_bands=64, hidden_dim=128, n_layers=3):
        super().__init__()
        self.n_stems    = n_stems
        self.n_fft      = n_fft
        self.hop        = hop
        self.n_bands    = n_bands
        self.hidden_dim = hidden_dim
        self.freq_bins  = n_fft // 2 + 1
        self.band_size  = self.freq_bins // n_bands + 1

        self.band_rnn = nn.ModuleList([
            nn.LSTM(
                input_size   = self.band_size * 2,
                hidden_size  = hidden_dim,
                num_layers   = n_layers,
                batch_first  = True,
                bidirectional= True,
                dropout      = 0.1 if n_layers > 1 else 0,
            )
            for _ in range(n_bands)
        ])

        self.global_rnn = nn.LSTM(
            input_size   = n_bands * hidden_dim * 2,
            hidden_size  = hidden_dim * 4,
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout      = 0.1,
        )

        # Mask decoder: linear -> linear (NO final Tanh squeeze to near-0).
        # We apply Tanh ourselves after a learnable scale so the model can't
        # collapse masks toward 0 just by shrinking the last linear layer's
        # weights uniformly - the gain is a free parameter that pushes back.
        self.mask_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, self.freq_bins * 2),
            )
            for _ in range(n_stems)
        ])
        # Learnable per-stem gain, initialized at 1.0 (not 0) so masks
        # start in a usable [-1,1]-ish range from epoch 1, not collapsed.
        self.mask_gain = nn.Parameter(torch.ones(n_stems))

    def forward(self, x):
        batch = x.shape[0]

        window = torch.hann_window(self.n_fft).to(x.device)
        stft   = torch.stft(x, self.n_fft, self.hop,
                            window=window, return_complex=True)
        freq_bins   = stft.shape[1]

        stft_real = stft.real
        stft_imag = stft.imag

        band_outputs = []
        for b in range(self.n_bands):
            f_start = b * (freq_bins // self.n_bands)
            f_end   = min(f_start + self.band_size, freq_bins)
            actual  = f_end - f_start
            pad     = self.band_size - actual

            band_r = stft_real[:, f_start:f_end, :]
            band_i = stft_imag[:, f_start:f_end, :]

            if pad > 0:
                band_r = torch.nn.functional.pad(band_r, (0, 0, 0, pad))
                band_i = torch.nn.functional.pad(band_i, (0, 0, 0, pad))

            band_feat = torch.cat([band_r, band_i], dim=1)
            band_feat = band_feat.permute(0, 2, 1)

            out, _ = self.band_rnn[b](band_feat)
            band_outputs.append(out)

        global_feat   = torch.cat(band_outputs, dim=-1)
        global_out, _ = self.global_rnn(global_feat)

        stems_stft = []
        for s in range(self.n_stems):
            mask_flat = self.mask_decoder[s](global_out)
            # Apply learnable gain BEFORE tanh so the model can scale
            # its own dynamic range instead of being stuck near zero.
            mask_flat = torch.tanh(mask_flat * self.mask_gain[s])

            mask_r = mask_flat[:, :, :freq_bins].permute(0, 2, 1)
            mask_i = mask_flat[:, :, freq_bins:].permute(0, 2, 1)
            mask_r = mask_r[:, :freq_bins, :]
            mask_i = mask_i[:, :freq_bins, :]

            stem_r = stft_real * mask_r - stft_imag * mask_i
            stem_i = stft_real * mask_i + stft_imag * mask_r

            stem_stft = torch.complex(stem_r, stem_i)
            stems_stft.append(stem_stft)

        stems_audio = []
        for s in range(self.n_stems):
            audio = torch.istft(
                stems_stft[s], self.n_fft, self.hop,
                window=window, length=x.shape[-1]
            )
            stems_audio.append(audio)

        return torch.stack(stems_audio, dim=1)


# ── Loss function (FIXED) ──────────────────────────────────────────────────────

def masked_loss(pred, target, active, eps=1e-8):
    """
    pred, target: (batch, n_stems, n_samples)
    active:       (batch, n_stems) - 1.0 if that stem is present in this mix, else 0.0

    For ACTIVE stems: standard L1 + spectral magnitude loss (stable, no division
    blowups), pushing pred toward target.
    For ABSENT stems: small L1-to-zero penalty only (don't let the model "cheat"
    by going silent everywhere, but don't let absent-stem loss dominate either).
    """
    active_mask = active.unsqueeze(-1)            # (batch, n_stems, 1)
    inactive_mask = 1.0 - active_mask

    # --- Active stems: reconstruct properly ---
    l1_active = torch.abs(pred - target) * active_mask
    l1_active = l1_active.sum() / (active_mask.sum() * pred.shape[-1] + eps)

    # Spectral magnitude loss for active stems (more perceptually meaningful
    # than raw waveform L1, and numerically stable - no division by target energy)
    pred_spec   = torch.stft(pred.reshape(-1, pred.shape[-1]), 1024, 256,
                             window=torch.hann_window(1024).to(pred.device),
                             return_complex=True).abs()
    target_spec = torch.stft(target.reshape(-1, target.shape[-1]), 1024, 256,
                             window=torch.hann_window(1024).to(target.device),
                             return_complex=True).abs()
    active_flat = active_mask.reshape(-1, 1, 1).expand_as(pred_spec)
    spec_loss   = (torch.abs(pred_spec - target_spec) * active_flat).sum() / \
                  (active_flat.sum() + eps)

    # --- Absent stems: light penalty for non-zero output (anti-leak), capped ---
    l1_inactive = torch.abs(pred) * inactive_mask
    l1_inactive = l1_inactive.sum() / (inactive_mask.sum() * pred.shape[-1] + eps)

    # Weighted combination - active reconstruction dominates
    loss = (1.0 * l1_active) + (0.5 * spec_loss) + (0.05 * l1_inactive)
    return loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print("=" * 60)
    print("Raga Spatial — 8-Stem Separator Training (FIXED v2)")
    print("Device: " + DEVICE)
    print("=" * 60)

    mix_dirs = sorted([
        os.path.join(MIXES_DIR, d)
        for d in os.listdir(MIXES_DIR)
        if os.path.isdir(os.path.join(MIXES_DIR, d))
        and os.path.exists(os.path.join(MIXES_DIR, d, "mix.wav"))
    ])

    if not mix_dirs:
        print("ERROR: No mixes found in " + MIXES_DIR)
        print("Run: python augment_mix.py --tier 1")
        return

    print("Total mixes: " + str(len(mix_dirs)))

    # Seeded shuffle of a sorted list: the same mixes land in val every run,
    # so pausing and resuming never leaks validation mixes into training.
    random.Random(SPLIT_SEED).shuffle(mix_dirs)
    val_size   = max(1, int(len(mix_dirs) * VAL_SPLIT))
    val_dirs   = mix_dirs[:val_size]
    train_dirs = mix_dirs[val_size:]

    print("Train: " + str(len(train_dirs)) + " | Val: " + str(len(val_dirs)))

    train_loader = DataLoader(StemDataset(train_dirs), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(StemDataset(val_dirs),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print("\nInitializing BSRNN model (v2 - gain-corrected)...")
    model = BandSplitRNN(n_stems=N_STEMS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(round(n_params / 1e6, 1)) + "M")

    if DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("GPU VRAM: " + str(round(vram_gb, 1)) + "GB")

    # NOTE: deliberately NOT resuming from old v1 checkpoint -
    # that model converged to a silent-output collapse and is not salvageable.
    checkpoint_path = os.path.join(MODELS_DIR, "separator_v2_latest.pt")
    start_epoch     = 0
    best_val_loss   = float("inf")

    if os.path.exists(checkpoint_path):
        print("Loading v2 checkpoint: " + checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print("Resuming v2 from epoch " + str(start_epoch))
    else:
        print("Starting fresh v2 training (old v1 checkpoints discarded - "
              "they converged to silent collapse)")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    # Keep earlier epochs in the log when resuming (v1 lost epochs 1-71 this way)
    log_path = os.path.join(LOGS_DIR, "training_log_v2.json")
    log = []
    if start_epoch > 0 and os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log = [e for e in json.load(f) if e.get("epoch", 0) <= start_epoch]
            print("Loaded " + str(len(log)) + " earlier epochs from the log")
        except (ValueError, OSError):
            print("WARNING: could not read existing log, starting a new one")

    print("\nStarting training...")
    print("Batch size: " + str(BATCH_SIZE) + " | Epochs: " + str(N_EPOCHS))
    print("-" * 60)

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        train_losses = []

        for mix, targets, active in tqdm(train_loader,
                                  desc="Epoch " + str(epoch+1) + "/" + str(N_EPOCHS),
                                  leave=False):
            mix     = mix.to(DEVICE)
            targets = targets.to(DEVICE)
            active  = active.to(DEVICE)

            optimizer.zero_grad()
            preds = model(mix)
            loss  = masked_loss(preds, targets, active)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        # Track avg output RMS to catch collapse early
        avg_pred_rms = []

        with torch.no_grad():
            for mix, targets, active in val_loader:
                mix     = mix.to(DEVICE)
                targets = targets.to(DEVICE)
                active  = active.to(DEVICE)
                preds   = model(mix)
                loss    = masked_loss(preds, targets, active)
                val_losses.append(loss.item())
                avg_pred_rms.append(torch.sqrt(torch.mean(preds**2)).item())

        val_loss = np.mean(val_losses)
        pred_rms = np.mean(avg_pred_rms)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        collapse_warning = "  <-- WARNING: low output energy, possible collapse" \
                            if pred_rms < 0.01 else ""

        print("Epoch " + str(epoch+1).rjust(3)
              + " | train: " + str(round(train_loss, 4)).rjust(8)
              + " | val: "   + str(round(val_loss,   4)).rjust(8)
              + " | pred_rms: " + str(round(pred_rms, 5)).rjust(8)
              + " | lr: "    + str(round(lr_now, 6))
              + collapse_warning)

        log.append({
            "epoch": epoch+1,
            "train_loss": round(float(train_loss), 4),
            "val_loss":   round(float(val_loss),   4),
            "pred_rms":   round(float(pred_rms), 6),
            "lr":         round(float(lr_now), 6),
        })
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        # Decide "best" BEFORE saving latest, so a resume sees the right value
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "categories":    CATEGORIES,
        }, checkpoint_path)

        if is_best:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "val_loss":   val_loss,
                "categories": CATEGORIES,
            }, os.path.join(MODELS_DIR, "separator_v2_best.pt"))
            print("  --> New best v2 model saved (val_loss=" + str(round(val_loss, 4)) + ")")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "val_loss":   val_loss,
                "categories": CATEGORIES,
            }, os.path.join(MODELS_DIR, "separator_v2_epoch" + str(epoch+1) + ".pt"))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE (v2)")
    print("Best val loss: " + str(round(best_val_loss, 4)))
    print("Best model: " + os.path.join(MODELS_DIR, "separator_v2_best.pt"))
    print("=" * 60)


if __name__ == "__main__":
    train()
