# Raga Spatial - train_separator.py
# Trains 8-stem BSRNN separator on synthetic mixes
# Saves checkpoints to D:\raga-spatial-data\models\

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

# Paths
MIXES_DIR  = r"D:\raga-spatial-data\synthetic_mixes"
MODELS_DIR = r"D:\raga-spatial-data\models"
LOGS_DIR   = r"D:\raga-spatial-data\logs"

# Training settings
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE  = 16000
DURATION     = 10.0
N_SAMPLES    = int(SAMPLE_RATE * DURATION)
N_STEMS      = 8
BATCH_SIZE   = 4      # small batch for 6GB VRAM
N_EPOCHS     = 100
LR           = 1e-3
LR_DECAY     = 0.98
SAVE_EVERY   = 10     # save checkpoint every N epochs
VAL_SPLIT    = 0.1    # 10% validation

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]

# ── Dataset ───────────────────────────────────────────────────────────────────

class StemDataset(Dataset):
    def __init__(self, mix_dirs, augment=True):
        self.mix_dirs = mix_dirs
        self.augment  = augment

    def __len__(self):
        return len(self.mix_dirs)

    def __getitem__(self, idx):
        mix_dir = self.mix_dirs[idx]

        # Load mix
        mix_path = os.path.join(mix_dir, "mix.wav")
        mix, _   = sf.read(mix_path)
        mix      = mix.astype(np.float32)
        if len(mix) < N_SAMPLES:
            mix = np.pad(mix, (0, N_SAMPLES - len(mix)))
        mix = mix[:N_SAMPLES]

        # Load target stems
        targets = []
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
            targets.append(stem)

        targets = np.stack(targets, axis=0)   # (8, N_SAMPLES)

        return torch.tensor(mix), torch.tensor(targets)


# ── Model: Lightweight BSRNN ──────────────────────────────────────────────────

class BandSplitRNN(nn.Module):
    """
    Simplified Band-Split RNN for 8-stem separation.
    Designed to fit in 6GB VRAM with batch_size=4.

    Architecture:
    1. STFT -> complex spectrogram
    2. Band split -> divide frequency into bands
    3. Band-wise BiLSTM -> process each band independently
    4. MLP decoder -> reconstruct masks per stem
    5. Apply masks -> separate stems
    6. ISTFT -> back to waveform
    """

    def __init__(self, n_stems=8, n_fft=1024, hop=256,
                 n_bands=64, hidden_dim=128, n_layers=3):
        super().__init__()
        self.n_stems    = n_stems
        self.n_fft      = n_fft
        self.hop        = hop
        self.n_bands    = n_bands
        self.hidden_dim = hidden_dim
        self.freq_bins  = n_fft // 2 + 1   # 513
        self.band_size  = self.freq_bins // n_bands + 1

        # Band-wise BiLSTM
        self.band_rnn = nn.ModuleList([
            nn.LSTM(
                input_size   = self.band_size * 2,  # real + imag
                hidden_size  = hidden_dim,
                num_layers   = n_layers,
                batch_first  = True,
                bidirectional= True,
                dropout      = 0.1 if n_layers > 1 else 0,
            )
            for _ in range(n_bands)
        ])

        # Global temporal RNN (across all bands)
        self.global_rnn = nn.LSTM(
            input_size   = n_bands * hidden_dim * 2,
            hidden_size  = hidden_dim * 4,
            num_layers   = 2,
            batch_first  = True,
            bidirectional= True,
            dropout      = 0.1,
        )

        # Mask decoder per stem
        self.mask_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, self.freq_bins * 2),  # real + imag mask
                nn.Tanh(),
            )
            for _ in range(n_stems)
        ])

    def forward(self, x):
        """
        x: (batch, n_samples) mono waveform
        returns: (batch, n_stems, n_samples)
        """
        batch = x.shape[0]

        # STFT
        window = torch.hann_window(self.n_fft).to(x.device)
        stft   = torch.stft(x, self.n_fft, self.hop,
                            window=window, return_complex=True)
        # stft: (batch, freq_bins, time_frames)
        freq_bins   = stft.shape[1]
        time_frames = stft.shape[2]

        # Real and imaginary parts
        stft_real = stft.real   # (batch, freq, time)
        stft_imag = stft.imag

        # Band split
        band_outputs = []
        for b in range(self.n_bands):
            f_start = b * (freq_bins // self.n_bands)
            f_end   = min(f_start + self.band_size, freq_bins)
            actual  = f_end - f_start
            pad     = self.band_size - actual

            band_r = stft_real[:, f_start:f_end, :]  # (batch, band_size, time)
            band_i = stft_imag[:, f_start:f_end, :]

            if pad > 0:
                band_r = torch.nn.functional.pad(band_r, (0, 0, 0, pad))
                band_i = torch.nn.functional.pad(band_i, (0, 0, 0, pad))

            # Interleave real and imag: (batch, time, band_size*2)
            band_feat = torch.cat([band_r, band_i], dim=1)
            band_feat = band_feat.permute(0, 2, 1)

            # BiLSTM for this band
            out, _ = self.band_rnn[b](band_feat)
            band_outputs.append(out)  # (batch, time, hidden*2)

        # Stack band outputs: (batch, time, n_bands * hidden*2)
        global_feat = torch.cat(band_outputs, dim=-1)

        # Global temporal RNN
        global_out, _ = self.global_rnn(global_feat)
        # global_out: (batch, time, hidden*8)

        # Decode mask per stem
        stems_stft = []
        for s in range(self.n_stems):
            mask_flat = self.mask_decoder[s](global_out)
            # mask_flat: (batch, time, freq_bins*2)
            mask_r = mask_flat[:, :, :freq_bins].permute(0, 2, 1)
            mask_i = mask_flat[:, :, freq_bins:].permute(0, 2, 1)

            # Trim to actual freq bins
            mask_r = mask_r[:, :freq_bins, :]
            mask_i = mask_i[:, :freq_bins, :]

            # Apply mask to input STFT
            stem_r = stft_real * mask_r - stft_imag * mask_i
            stem_i = stft_real * mask_i + stft_imag * mask_r

            stem_stft = torch.complex(stem_r, stem_i)
            stems_stft.append(stem_stft)

        # ISTFT per stem
        stems_audio = []
        for s in range(self.n_stems):
            audio = torch.istft(
                stems_stft[s], self.n_fft, self.hop,
                window=window, length=x.shape[-1]
            )
            stems_audio.append(audio)

        # Stack: (batch, n_stems, n_samples)
        return torch.stack(stems_audio, dim=1)


# ── Loss function ─────────────────────────────────────────────────────────────

def sdr_loss(pred, target, eps=1e-8):
    """
    Signal-to-Distortion Ratio loss.
    Higher SDR = better separation.
    We minimize negative SDR.
    pred, target: (batch, n_stems, n_samples)
    """
    dot        = (pred * target).sum(dim=-1)
    target_pow = (target ** 2).sum(dim=-1)
    noise      = pred - dot.unsqueeze(-1) / (target_pow.unsqueeze(-1) + eps) * target
    noise_pow  = (noise ** 2).sum(dim=-1)
    sdr        = 10 * torch.log10(target_pow / (noise_pow + eps) + eps)
    return -sdr.mean()


def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def combined_loss(pred, target):
    return sdr_loss(pred, target) + 0.1 * l1_loss(pred, target)


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print("=" * 60)
    print("Raga Spatial — 8-Stem Separator Training")
    print("Device: " + DEVICE)
    print("=" * 60)

    # Find all mix directories
    mix_dirs = sorted([
        os.path.join(MIXES_DIR, d)
        for d in os.listdir(MIXES_DIR)
        if os.path.isdir(os.path.join(MIXES_DIR, d))
        and os.path.exists(os.path.join(MIXES_DIR, d, "mix.wav"))
    ])

    if not mix_dirs:
        print("ERROR: No mixes found in " + MIXES_DIR)
        print("Run augment_mix.py first")
        return

    print("Total mixes: " + str(len(mix_dirs)))

    # Train/val split
    random.shuffle(mix_dirs)
    val_size   = max(1, int(len(mix_dirs) * VAL_SPLIT))
    val_dirs   = mix_dirs[:val_size]
    train_dirs = mix_dirs[val_size:]

    print("Train: " + str(len(train_dirs)) + " | Val: " + str(len(val_dirs)))

    # Datasets
    train_dataset = StemDataset(train_dirs, augment=True)
    val_dataset   = StemDataset(val_dirs,   augment=False)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    # Model
    print("\nInitializing BSRNN model...")
    model = BandSplitRNN(n_stems=N_STEMS).to(DEVICE)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(round(n_params / 1e6, 1)) + "M")

    # Check VRAM
    if DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("GPU VRAM: " + str(round(vram_gb, 1)) + "GB")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(MODELS_DIR, "separator_latest.pt")
    start_epoch     = 0
    best_val_loss   = float("inf")

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint: " + checkpoint_path)
        ckpt        = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print("Resuming from epoch " + str(start_epoch))

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    if os.path.exists(checkpoint_path) and "optimizer" in torch.load(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        optimizer.load_state_dict(ckpt["optimizer"])

    # Training log
    log = []

    print("\nStarting training...")
    print("Batch size: " + str(BATCH_SIZE) + " | Epochs: " + str(N_EPOCHS))
    print("-" * 60)

    for epoch in range(start_epoch, N_EPOCHS):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_losses = []

        for mix, targets in tqdm(train_loader,
                                  desc="Epoch " + str(epoch+1) + "/" + str(N_EPOCHS),
                                  leave=False):
            mix     = mix.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(mix)
            loss  = combined_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for mix, targets in val_loader:
                mix     = mix.to(DEVICE)
                targets = targets.to(DEVICE)
                preds   = model(mix)
                loss    = combined_loss(preds, targets)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print("Epoch " + str(epoch+1).rjust(3)
              + " | train: " + str(round(train_loss, 4)).rjust(8)
              + " | val: "   + str(round(val_loss,   4)).rjust(8)
              + " | lr: "    + str(round(lr_now, 6)))

        # Log
        log.append({
            "epoch": epoch+1,
            "train_loss": round(float(train_loss), 4),
            "val_loss":   round(float(val_loss),   4),
            "lr":         round(float(lr_now), 6),
        })
        with open(os.path.join(LOGS_DIR, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        # Save latest checkpoint
        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "categories":    CATEGORIES,
        }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "val_loss":   val_loss,
                "categories": CATEGORIES,
            }, os.path.join(MODELS_DIR, "separator_best.pt"))
            print("  --> New best model saved (val_loss=" + str(round(val_loss, 4)) + ")")

        # Save periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "val_loss":   val_loss,
                "categories": CATEGORIES,
            }, os.path.join(MODELS_DIR, "separator_epoch" + str(epoch+1) + ".pt"))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("Best val loss: " + str(round(best_val_loss, 4)))
    print("Best model: " + os.path.join(MODELS_DIR, "separator_best.pt"))
    print("=" * 60)


if __name__ == "__main__":
    train()
