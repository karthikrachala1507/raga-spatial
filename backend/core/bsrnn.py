# Raga Spatial - core/bsrnn.py
# The 8-stem separator model. Used by training (training/train_separator.py)
# and by the app (core/separator_8stem.py).
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

import torch
import torch.nn as nn

BAND_LAYOUT = [(4, 16), (8, 8), (16, 8), (32, 8)]    # (bins per band, number of bands) = 40 bands
DEFAULT_FEATURE_DIM = 112    # width of each band's feature vector
DEFAULT_N_BLOCKS    = 8      # time/band LSTM blocks


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
    def __init__(self, n_stems=8, n_fft=1024, hop=256,
                 feature_dim=DEFAULT_FEATURE_DIM, n_blocks=DEFAULT_N_BLOCKS):
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
