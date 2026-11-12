# Debug script - check why stems are silent
# Run from backend/ folder: python debug_eval.py

import os, sys
import numpy as np
import torch
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_separator import BandSplitRNN

MODEL_PATH = r"D:\raga-spatial-data\models\separator_best.pt"
SONG_PATH  = r"D:\raga-spatial-data\real_songs\test5.mp3"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print("=== Loading model ===")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
print("Checkpoint keys:", list(ckpt.keys()))
print("Epoch:", ckpt.get("epoch"))

model = BandSplitRNN(n_stems=8).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

print("\n=== Checking model weights for NaN/zero ===")
total_params = 0
zero_params  = 0
nan_params   = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    zero_params  += (param == 0).sum().item()
    nan_params   += torch.isnan(param).sum().item()
print("Total params:", total_params)
print("Zero params:", zero_params, "(", round(zero_params/total_params*100,2), "%)")
print("NaN params:", nan_params)

print("\n=== Loading 10 seconds of test song ===")
audio, sr = librosa.load(SONG_PATH, sr=16000, mono=True, duration=10.0)
print("Audio shape:", audio.shape, "SR:", sr)
print("Audio RMS:", np.sqrt(np.mean(audio**2)))
print("Audio max abs:", np.max(np.abs(audio)))

print("\n=== Running model forward pass ===")
chunk_samples = int(16000 * 10.0)
if len(audio) < chunk_samples:
    audio = np.pad(audio, (0, chunk_samples - len(audio)))
audio = audio[:chunk_samples]

x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
print("Input tensor shape:", x.shape)
print("Input tensor RMS:", torch.sqrt(torch.mean(x**2)).item())

with torch.no_grad():
    out = model(x)

print("\nOutput tensor shape:", out.shape)
print("Output contains NaN:", torch.isnan(out).any().item())
print("Output contains Inf:", torch.isinf(out).any().item())
print("Output max abs:", torch.max(torch.abs(out)).item())
print("Output min:", torch.min(out).item())
print("Output max:", torch.max(out).item())

print("\n=== Per-stem output stats ===")
out_np = out.squeeze(0).cpu().numpy()
categories = ["percussion","plucked_strings","bowed_strings","wind",
              "keys_synth","vocals","bass","folk_texture"]
for i, cat in enumerate(categories):
    stem = out_np[i]
    print(cat.ljust(20), "RMS:", round(float(np.sqrt(np.mean(stem**2))), 8),
          "max:", round(float(np.max(np.abs(stem))), 8))
