# Raga Spatial - train_separator.py (v3: standard band-split RNN)
# Trains the 8-stem separator.
#
# History: v1 collapsed to silence (its SDR loss rewarded silence). v2 fixed
# the loss but its model learned only bass in 4 epochs. v3 keeps the v2 loss
# and data pipeline and replaces the model with the standard band-split RNN,
# which in an 11-minute trial beat 5 hours of the v2 model on every stem.
#
# Loss: L1 + spectral-magnitude loss on active stems, light penalty on absent ones.
#
# Data: mixes are built ON THE FLY from the stem-clip pools, so every training
# example is a new combination and new datasets are picked up at the start of
# the next epoch - no mix regeneration, no disk used for mixes.
#   - target stems are exactly what goes into the mix (stems sum to the mix)
#   - clips are drawn per SOURCE FILE, not per clip, so a category with a few
#     long recordings no longer drowns out the others
#   - 10% of source files are held out and never trained on; validation uses
#     a fixed, seeded set of mixes built only from those held-out sources
#   - "best" is chosen by separation quality in dB (val_sdr), not by the loss
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
MIXES_PER_EPOCH = 12000  # training mixes drawn per epoch (all freshly generated)
N_VAL_MIXES  = 300       # fixed validation mixes from held-out sources
N_SDR_MIXES  = 80        # of those, how many are scored in dB each epoch
FEATURE_DIM  = 112       # width of each band's feature vector
N_BLOCKS     = 8         # time/band LSTM blocks
CROP_SECONDS = 6.0       # each training step sees a random 6 s window of a 10 s mix
LR           = 1e-3
LR_DECAY     = 0.98
SAVE_EVERY   = 10
SPLIT_SEED   = 1507      # fixes the validation set across runs and resumes
VAL_HOLDOUT_MOD = 10     # 1 in 10 source files is held out for validation
USE_AMP      = True      # mixed precision: about half the GPU memory
NUM_WORKERS  = 4         # background processes that build mixes (0 = none)
# Tier 2 sources with these prefixes are still being added, so they are kept
# out of the validation set to keep it identical from run to run.
VAL_EXCLUDE_PREFIXES = ("slakh2100__",)
CHECKPOINT_NAME = "separator_v3"
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
                            # held out everywhere; datasets still growing stay
                            # out of the fixed validation set
                            if not source.startswith(VAL_EXCLUDE_PREFIXES):
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


# ── Model: band-split RNN (BSRNN) ─────────────────────────────────────────────
#
# Standard band-split design (Luo & Yu, 2022), sized for a 6 GB GPU:
#   1. the spectrogram is cut into 40 bands, narrow at low frequencies and wide
#      at high ones; each band is normalised and projected to FEATURE_DIM numbers
#   2. a stack of blocks alternates an LSTM along TIME (how each band evolves)
#      with an LSTM across BANDS (how the bands relate at one moment, which is
#      what tells a flute from a violin)
#   3. every band keeps its own feature vector to the end and predicts its own
#      complex mask for each of the 8 stems
#
# The earlier model squeezed all 513 bins x 8 stems of every frame through one
# 1024-number vector and was not separating after 4 epochs (about 0 dB). This
# one has ~20x fewer parameters and no such bottleneck.

BAND_LAYOUT = [(4, 16), (8, 8), (16, 8), (32, 8)]    # (bins per band, number of bands) = 40 bands


def band_edges(freq_bins):
    edges, start = [], 0
    for width, count in BAND_LAYOUT:
        for _ in range(count):
            edges.append((start, start + width))
            start += width
    edges[-1] = (edges[-1][0], freq_bins)              # last band takes the top bin
    return edges


class _RNNBlock(nn.Module):
    """Normalise, run a bidirectional LSTM along one axis, add back (residual)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.rnn  = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(dim * 2, dim)

    def forward(self, x):                              # (sequences, length, dim)
        y = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y, _ = self.rnn(y)
        return x + self.fc(y)


class _DualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.along_time  = _RNNBlock(dim)
        self.along_bands = _RNNBlock(dim)

    def forward(self, h):                              # (batch, bands, frames, dim)
        b, k, t, n = h.shape
        h = self.along_time(h.reshape(b * k, t, n)).reshape(b, k, t, n)
        h = h.transpose(1, 2)                          # (batch, frames, bands, dim)
        h = self.along_bands(h.reshape(b * t, k, n)).reshape(b, t, k, n)
        return h.transpose(1, 2)


class BandSplitRNN(nn.Module):
    def __init__(self, n_stems=8, n_fft=1024, hop=256, feature_dim=96, n_blocks=8):
        super().__init__()
        self.n_stems   = n_stems
        self.n_fft     = n_fft
        self.hop       = hop
        self.freq_bins = n_fft // 2 + 1
        self.bands     = band_edges(self.freq_bins)
        widths         = [e - s for s, e in self.bands]

        self.split_norm = nn.ModuleList([nn.GroupNorm(1, 2 * w) for w in widths])
        self.split_fc   = nn.ModuleList([nn.Linear(2 * w, feature_dim) for w in widths])
        self.blocks     = nn.ModuleList([_DualBlock(feature_dim) for _ in range(n_blocks)])
        self.mask_norm  = nn.ModuleList([nn.GroupNorm(1, feature_dim) for _ in widths])
        self.mask_mlp   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.Tanh(),
                nn.Linear(feature_dim * 4, 2 * w * n_stems),
            )
            for w in widths
        ])

    def forward(self, x):                              # (batch, samples)
        n_samples = x.shape[-1]
        device_type = "cuda" if x.is_cuda else "cpu"

        # Band split in full precision: raw spectrogram values are too large
        # for half-precision normalisation.
        with torch.autocast(device_type=device_type, enabled=False):
            x      = x.float()
            window = torch.hann_window(self.n_fft, device=x.device)
            stft   = torch.stft(x, self.n_fft, self.hop, window=window, return_complex=True)
            feats  = []
            for i, (s, e) in enumerate(self.bands):
                band = torch.cat([stft.real[:, s:e], stft.imag[:, s:e]], dim=1)   # (b, 2w, t)
                band = self.split_norm[i](band).transpose(1, 2)                   # (b, t, 2w)
                feats.append(self.split_fc[i](band))                              # (b, t, dim)
            h = torch.stack(feats, dim=1)                                         # (b, k, t, dim)

        for block in self.blocks:
            h = block(h)

        batch, _, frames, _ = h.shape
        masks = []
        for i, (s, e) in enumerate(self.bands):
            m = self.mask_norm[i](h[:, i].transpose(1, 2)).transpose(1, 2)
            m = self.mask_mlp[i](m)                                               # (b, t, 2*w*stems)
            m = m.reshape(batch, frames, self.n_stems, 2, e - s)
            masks.append(m.permute(0, 2, 4, 1, 3))                                # (b, stems, w, t, 2)
        mask = torch.cat(masks, dim=2).float()                                    # (b, stems, bins, t, 2)

        with torch.autocast(device_type=device_type, enabled=False):
            mask = torch.complex(mask[..., 0], mask[..., 1])
            est  = mask * stft[:, None]                                           # (b, stems, bins, t)
            audio = torch.istft(est.reshape(batch * self.n_stems, self.freq_bins, frames),
                                self.n_fft, self.hop, window=window, length=n_samples)
        return audio.reshape(batch, self.n_stems, n_samples)


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
    crop = int(CROP_SECONDS * SAMPLE_RATE)
    if crop < mix.shape[-1]:
        start   = random.randint(0, mix.shape[-1] - crop)
        mix     = mix[:, start:start + crop]
        targets = targets[:, :, start:start + crop]
        active  = (targets.pow(2).mean(dim=-1).sqrt() > MIN_TARGET_RMS).float()
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


def _sdr_db(target, estimate):
    noise = target - estimate
    return 10.0 * torch.log10((target.pow(2).sum() + 1e-10) / (noise.pow(2).sum() + 1e-10))


def stem_sdrs(mix, preds, targets, active):
    """SDR per active stem, scored the way the app uses the model: its outputs
    become soft masks (summing to 1) on the mix. 0 dB = no better than silence."""
    window = torch.hann_window(1024, device=mix.device)
    mix_tf = torch.stft(mix, 1024, 256, window=window, return_complex=True)
    power  = torch.stft(preds, 1024, 256, window=window, return_complex=True).abs() ** 2
    masks  = power / (power.sum(dim=0, keepdim=True) + 1e-10)
    est    = torch.istft(mix_tf[None] * masks, 1024, 256, window=window,
                         length=mix.shape[-1])
    return {s: float(_sdr_db(targets[s], est[s]))
            for s in range(preds.shape[0]) if active[s] > 0.5}


def evaluate(model, loader, use_amp):
    """Validation loss, output RMS, output level relative to targets, and
    separation quality in dB (overall median and per category)."""
    model.eval()
    losses, pred_rms_all, pred_e, tgt_e = [], [], 0.0, 0.0
    sdrs, scored = {c: [] for c in CATEGORIES}, 0
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
            for b in range(mix.shape[0]):
                if scored < N_SDR_MIXES:
                    for s, v in stem_sdrs(mix[b], preds[b], targets[b], active[b]).items():
                        sdrs[CATEGORIES[s]].append(v)
                    scored += 1
    level_ratio = (pred_e / tgt_e) ** 0.5 if tgt_e > 0 else 0.0
    per_cat = {c: round(float(np.median(v)), 2) for c, v in sdrs.items() if v}
    all_sdr = [x for v in sdrs.values() for x in v]
    val_sdr = float(np.median(all_sdr)) if all_sdr else 0.0
    return float(np.mean(losses)), float(np.mean(pred_rms_all)), level_ratio, val_sdr, per_cat


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print("=" * 60)
    print("Raga Spatial - 8-Stem Separator Training (v3 band-split RNN)")
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

    print("\nInitializing band-split RNN (dim " + str(FEATURE_DIM)
          + ", " + str(N_BLOCKS) + " blocks)...")
    model = BandSplitRNN(n_stems=N_STEMS, feature_dim=FEATURE_DIM,
                         n_blocks=N_BLOCKS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameters: " + str(round(n_params / 1e6, 1)) + "M")
    if DEVICE == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("GPU VRAM: " + str(round(vram_gb, 1)) + "GB")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE == "cuda")

    # v1 / v2 checkpoints are a different model and are never loaded.
    model_config    = {"feature_dim": FEATURE_DIM, "n_blocks": N_BLOCKS}
    checkpoint_path = os.path.join(MODELS_DIR, CHECKPOINT_NAME + "_latest.pt")
    best_path       = os.path.join(MODELS_DIR, CHECKPOINT_NAME + "_best.pt")
    log_path        = os.path.join(LOGS_DIR, "training_log_v3.json")
    start_epoch     = 0
    best_val_sdr    = float("-inf")
    log             = []

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint: " + checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and USE_AMP:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_sdr  = ckpt.get("best_val_sdr", float("-inf"))
        del ckpt
        # Keep earlier epochs in the log when resuming (v1 lost epochs 1-71 this way)
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = [e for e in json.load(f) if e.get("epoch", 0) <= start_epoch]
            except (ValueError, OSError):
                print("WARNING: could not read existing log, starting a new one")
        print("Resuming from epoch " + str(start_epoch + 1)
              + " (" + str(len(log)) + " earlier epochs in the log)")
    else:
        print("Starting fresh v3 training")

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

        val_loss, pred_rms, level_ratio, val_sdr, sdr_per_cat = evaluate(
            model, val_loader, USE_AMP)
        scheduler.step()

        lr_now  = optimizer.param_groups[0]["lr"]
        minutes = (time.time() - t_epoch) / 60
        collapse_warning = "  <-- WARNING: low output energy, possible collapse" \
                            if pred_rms < 0.005 else ""

        print("Epoch " + str(epoch+1).rjust(3)
              + " | train: " + str(round(train_loss, 4)).rjust(8)
              + " | val: "   + str(round(val_loss,   4)).rjust(8)
              + " | SDR: "   + (str(round(val_sdr, 2)) + " dB").rjust(8)
              + " | level: " + str(round(level_ratio, 2)).rjust(5)
              + " | lr: "    + str(round(lr_now, 6))
              + " | " + str(round(minutes)) + " min"
              + (" | skipped " + str(skipped) if skipped else "")
              + collapse_warning)

        log.append({
            "epoch": epoch+1,
            "train_loss":  round(train_loss, 4),
            "val_loss":    round(val_loss,   4),
            "val_sdr_db":  round(val_sdr, 2),
            "sdr_per_stem": sdr_per_cat,
            "pred_rms":    round(pred_rms, 6),
            "level_ratio": round(level_ratio, 3),
            "lr":          round(float(lr_now), 6),
            "minutes":     round(minutes, 1),
            "train_sources": n_sources,
            "skipped_batches": skipped,
        })
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        print("          per stem (dB): " + ", ".join(
            c + " " + str(v) for c, v in sdr_per_cat.items()))

        # "Best" = best separation in dB. Decided BEFORE saving latest, so a
        # resume sees the right value.
        is_best = val_sdr > best_val_sdr
        if is_best:
            best_val_sdr = val_sdr

        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scaler":        scaler.state_dict(),
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "val_sdr":       val_sdr,
            "best_val_sdr":  best_val_sdr,
            "config":        model_config,
            "categories":    CATEGORIES,
        }, checkpoint_path + ".tmp")
        os.replace(checkpoint_path + ".tmp", checkpoint_path)   # never a half-written file

        if is_best:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "val_loss":   val_loss,
                "val_sdr":    val_sdr,
                "config":     model_config,
                "categories": CATEGORIES,
            }, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
            print("  --> New best model saved (SDR " + str(round(val_sdr, 2)) + " dB)")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "val_loss":   val_loss,
                "val_sdr":    val_sdr,
                "config":     model_config,
                "categories": CATEGORIES,
            }, os.path.join(MODELS_DIR, CHECKPOINT_NAME + "_epoch" + str(epoch+1) + ".pt"))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE (v3)")
    print("Best separation: " + str(round(best_val_sdr, 2)) + " dB")
    print("Best model: " + best_path)
    print("=" * 60)


if __name__ == "__main__":
    train()
