# Raga Spatial - train_separator.py (v2 model + loss, on-the-fly mixes)
# Trains the 8-stem BSRNN separator.
#
# Model and loss are the v2 design (fixes the v1 silent-output collapse):
#   - L1 + spectral-magnitude loss on active stems, light penalty on absent ones
#   - learnable mask gain, pred_rms collapse monitor
#
# Data: mixes are built ON THE FLY from the stem-clip pools, so every training
# example is a new combination and new datasets are picked up at the start of
# the next epoch - no mix regeneration, no disk used for mixes.
#   - target stems are exactly what goes into the mix (stems sum to the mix)
#   - clips are drawn per SOURCE FILE, not per clip, so a category with a few
#     long recordings no longer drowns out the others
#   - 10% of source files are held out and never trained on; validation uses
#     a fixed, seeded set of mixes built only from those held-out sources
#
# Run:    start_training.bat      (or: python train_separator.py)
# Pause:  Ctrl+C                  Resume: run it again

import os
import sys
import json
import zlib
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from tqdm import tqdm

# Stem-clip pools: each has one sub-folder per category with 10 s, 16 kHz mono WAVs.
# Tier 1 = FreeSounds + YouTube clips. Tier 2 = open datasets (prepare_datasets.py).
POOL_DIRS = [
    r"D:\raga-spatial-data\processed\stems",
    r"D:\raga-spatial-data\processed\stems_t2",
]
MODELS_DIR = r"D:\raga-spatial-data\models"
LOGS_DIR   = r"D:\raga-spatial-data\logs"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE  = 16000
DURATION     = 10.0
N_SAMPLES    = int(SAMPLE_RATE * DURATION)
N_STEMS      = 8
BATCH_SIZE   = 4
N_EPOCHS     = 100
MIXES_PER_EPOCH = 4500   # training mixes drawn per epoch (all freshly generated)
N_VAL_MIXES  = 400       # fixed validation mixes from held-out sources
LR           = 1e-3
LR_DECAY     = 0.98
SAVE_EVERY   = 10
SPLIT_SEED   = 1507      # fixes the validation set across runs and resumes
VAL_HOLDOUT_MOD = 10     # 1 in 10 source files is held out for validation
USE_AMP      = True      # mixed precision: 4.6 GB instead of 6.8 GB VRAM, ~17% faster
NUM_WORKERS  = 2         # background processes that build mixes (0 = none)
MIN_TARGET_RMS = 1e-4    # below this, a target stem is considered "absent"

STEMS_MIN, STEMS_MAX = 2, 6        # categories per mix
LEVEL_MIN, LEVEL_MAX = 0.21, 1.0   # level of each stem in the mix
SECOND_CLIP_PROB     = 0.25        # chance of layering 2 sources in one category
MAX_CLIPS_WEIGHT     = 30          # cap so very long recordings do not dominate

CATEGORIES = [
    "percussion", "plucked_strings", "bowed_strings",
    "wind", "keys_synth", "vocals", "bass", "folk_texture"
]

# ── Data: clip pools and on-the-fly mixing ───────────────────────────────────

def clip_source(filename):
    """Source-file id of a clip.
    Tier 1: percussion_00012_003.wav        -> percussion_00012
    Tier 2: slakh__Track00001_S03__0007.wav -> slakh__Track00001_S03"""
    stem = os.path.splitext(filename)[0]
    if "__" in stem:
        return stem.rsplit("__", 1)[0]
    return stem.rsplit("_", 1)[0]


def is_val_source(source):
    return zlib.crc32(source.encode("utf-8")) % VAL_HOLDOUT_MOD == 0


class ClipPool:
    """Index of all clips, grouped category -> source file -> clip paths."""

    def __init__(self, pool_dirs):
        self.pool_dirs = pool_dirs
        self.train = {}
        self.val   = {}
        self.refresh()

    def refresh(self):
        train = {c: {} for c in CATEGORIES}
        val   = {c: {} for c in CATEGORIES}
        for pool_idx, root in enumerate(self.pool_dirs):
            for cat in CATEGORIES:
                cat_dir = os.path.join(root, cat)
                if not os.path.isdir(cat_dir):
                    continue
                with os.scandir(cat_dir) as entries:
                    for e in entries:
                        if not e.name.endswith(".wav"):
                            continue
                        source = clip_source(e.name)
                        if is_val_source(source):
                            # held out everywhere; only pool 0 feeds the fixed val set
                            if pool_idx == 0:
                                val[cat].setdefault(source, []).append(e.path)
                        else:
                            train[cat].setdefault(source, []).append(e.path)
        self.train = self._index(train)
        self.val   = self._index(val)

    @staticmethod
    def _index(by_cat):
        index = {}
        for cat, sources in by_cat.items():
            names = sorted(sources)
            if not names:
                continue
            for n in names:
                sources[n].sort()
            weights = [min(len(sources[n]), MAX_CLIPS_WEIGHT) ** 0.5 for n in names]
            index[cat] = (names, weights, sources)
        return index

    @staticmethod
    def summary(index):
        return {cat: len(index[cat][0]) for cat in CATEGORIES if cat in index}


def sample_recipe(rng, index):
    """Pick which clips go into one mix and at what level."""
    cats = [c for c in CATEGORIES if c in index]
    n    = rng.randint(min(STEMS_MIN, len(cats)), min(STEMS_MAX, len(cats)))
    recipe = []
    for cat in rng.sample(cats, n):
        names, weights, sources = index[cat]
        picks = 2 if (rng.random() < SECOND_CLIP_PROB and len(names) > 1) else 1
        for source in rng.choices(names, weights=weights, k=picks):
            path  = rng.choice(sources[source])
            level = rng.uniform(LEVEL_MIN, LEVEL_MAX) / picks ** 0.5
            recipe.append((CATEGORIES.index(cat), path, level))
    return recipe


def load_clip(path):
    try:
        audio, _ = sf.read(path, dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    return audio[:N_SAMPLES]


def render_recipe(recipe):
    """Build (mix, targets, active). Targets are exactly what is in the mix."""
    targets = np.zeros((N_STEMS, N_SAMPLES), dtype=np.float32)
    for cat_idx, path, level in recipe:
        audio = load_clip(path)
        if audio is not None:
            targets[cat_idx] += audio * level
    mix  = targets.sum(axis=0)
    peak = max(float(np.abs(mix).max()), float(np.abs(targets).max()))
    if peak > 0.95:                       # one scale for the mix and every stem
        scale    = 0.95 / peak
        mix     *= scale
        targets *= scale
    rms    = np.sqrt((targets ** 2).mean(axis=1))
    active = (rms > MIN_TARGET_RMS).astype(np.float32)
    return torch.from_numpy(mix), torch.from_numpy(targets), torch.from_numpy(active)


class TrainMixDataset(Dataset):
    """Every item is a brand-new random mix from the training sources."""

    def __init__(self, pool, n_items):
        self.pool    = pool
        self.n_items = n_items
        self.rng     = None

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        if self.rng is None:
            # created on first use, so each DataLoader worker gets its own
            # randomness instead of a copy of the same generator
            self.rng = random.Random()
        return render_recipe(sample_recipe(self.rng, self.pool.train))


class ValMixDataset(Dataset):
    """A fixed list of mixes from held-out sources (same every epoch and resume)."""

    def __init__(self, pool, n_items):
        rng = random.Random(SPLIT_SEED)
        self.recipes = [sample_recipe(rng, pool.val) for _ in range(n_items)]

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        return render_recipe(self.recipes[idx])


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

def train_step(model, batch, optimizer, scaler, use_amp):
    """One optimisation step. Returns the loss value, or None if it was skipped."""
    mix, targets, active = [t.to(DEVICE, non_blocking=True) for t in batch]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16,
                        enabled=use_amp and DEVICE == "cuda"):
        preds = model(mix)
    loss = masked_loss(preds.float(), targets, active)
    if not torch.isfinite(loss):
        return None                       # skip a bad batch instead of poisoning weights
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def evaluate(model, loader, use_amp):
    """Validation loss, overall output RMS, and output level relative to targets."""
    model.eval()
    losses, pred_rms_all, pred_e, tgt_e = [], [], 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            mix, targets, active = [t.to(DEVICE) for t in batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp and DEVICE == "cuda"):
                preds = model(mix)
            preds = preds.float()
            losses.append(masked_loss(preds, targets, active).item())
            pred_rms_all.append(torch.sqrt(torch.mean(preds ** 2)).item())
            act     = active.unsqueeze(-1)
            pred_e += (preds ** 2 * act).sum().item()
            tgt_e  += (targets ** 2 * act).sum().item()
    level_ratio = (pred_e / tgt_e) ** 0.5 if tgt_e > 0 else 0.0
    return float(np.mean(losses)), float(np.mean(pred_rms_all)), level_ratio


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print("=" * 60)
    print("Raga Spatial - 8-Stem Separator Training (v2 model, on-the-fly mixes)")
    print("Device: " + DEVICE + " | mixed precision: " + str(USE_AMP and DEVICE == "cuda"))
    print("=" * 60)

    pool = ClipPool(POOL_DIRS)
    if len(pool.train) < 2 or len(pool.val) < 2:
        print("ERROR: not enough stem clips found in:")
        for d in POOL_DIRS:
            print("  " + d)
        return
    print("Training sources per category:   " + str(pool.summary(pool.train)))
    print("Validation sources per category: " + str(pool.summary(pool.val)))

    train_set = TrainMixDataset(pool, MIXES_PER_EPOCH)
    val_set   = ValMixDataset(pool, N_VAL_MIXES)
    # Workers build mixes while the GPU trains. They restart every epoch, which
    # is how they see the refreshed clip pool.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("Mixes per epoch: " + str(MIXES_PER_EPOCH) + " (new every epoch) | "
          "fixed validation mixes: " + str(len(val_set)))

    print("\nInitializing BSRNN model (v2 - gain-corrected)...")
    model = BandSplitRNN(n_stems=N_STEMS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(round(n_params / 1e6, 1)) + "M")
    if DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("GPU VRAM: " + str(round(vram_gb, 1)) + "GB")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE == "cuda")

    # v1 checkpoints are never loaded: that model collapsed to silence.
    checkpoint_path = os.path.join(MODELS_DIR, "separator_v2_latest.pt")
    log_path        = os.path.join(LOGS_DIR, "training_log_v2.json")
    start_epoch     = 0
    best_val_loss   = float("inf")
    log             = []

    if os.path.exists(checkpoint_path):
        print("Loading v2 checkpoint: " + checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and USE_AMP:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        del ckpt
        # Keep earlier epochs in the log when resuming (v1 lost epochs 1-71 this way)
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = [e for e in json.load(f) if e.get("epoch", 0) <= start_epoch]
            except (ValueError, OSError):
                print("WARNING: could not read existing log, starting a new one")
        print("Resuming v2 from epoch " + str(start_epoch + 1)
              + " (" + str(len(log)) + " earlier epochs in the log)")
    else:
        print("Starting fresh v2 training")

    print("\nStarting training...")
    print("Batch size: " + str(BATCH_SIZE) + " | Epochs: " + str(N_EPOCHS))
    print("-" * 60)

    for epoch in range(start_epoch, N_EPOCHS):
        t_epoch = time.time()
        if epoch > start_epoch:
            pool.refresh()                # pick up clips added since the last epoch
        n_sources = sum(pool.summary(pool.train).values())

        model.train()
        train_losses, skipped = [], 0
        for batch in tqdm(train_loader,
                          desc="Epoch " + str(epoch+1) + "/" + str(N_EPOCHS),
                          leave=False):
            loss = train_step(model, batch, optimizer, scaler, USE_AMP)
            if loss is None:
                skipped += 1
            else:
                train_losses.append(loss)

        if not train_losses:
            print("ERROR: every batch produced a non-finite loss. Stopping.")
            return
        train_loss = float(np.mean(train_losses))

        val_loss, pred_rms, level_ratio = evaluate(model, val_loader, USE_AMP)
        scheduler.step()

        lr_now  = optimizer.param_groups[0]["lr"]
        minutes = (time.time() - t_epoch) / 60
        collapse_warning = "  <-- WARNING: low output energy, possible collapse" \
                            if pred_rms < 0.01 else ""

        print("Epoch " + str(epoch+1).rjust(3)
              + " | train: " + str(round(train_loss, 4)).rjust(8)
              + " | val: "   + str(round(val_loss,   4)).rjust(8)
              + " | pred_rms: " + str(round(pred_rms, 5)).rjust(8)
              + " | level: " + str(round(level_ratio, 2)).rjust(5)
              + " | lr: "    + str(round(lr_now, 6))
              + " | " + str(round(minutes)) + " min"
              + (" | skipped " + str(skipped) if skipped else "")
              + collapse_warning)

        log.append({
            "epoch": epoch+1,
            "train_loss":  round(train_loss, 4),
            "val_loss":    round(val_loss,   4),
            "pred_rms":    round(pred_rms, 6),
            "level_ratio": round(level_ratio, 3),
            "lr":          round(float(lr_now), 6),
            "minutes":     round(minutes, 1),
            "train_sources": n_sources,
            "skipped_batches": skipped,
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
            "scaler":        scaler.state_dict(),
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "best_val_loss": best_val_loss,
            "categories":    CATEGORIES,
        }, checkpoint_path + ".tmp")
        os.replace(checkpoint_path + ".tmp", checkpoint_path)   # never a half-written file

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
