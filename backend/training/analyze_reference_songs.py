# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - analyze_reference_songs.py
# Measures what is characteristic about each reference song BEFORE anything is
# trained on it: how the mix is spread left/right per frequency band, how much
# of the energy is sub-bass, and what the 8-stem model finds in it (stem shares,
# instruments named by BEATs, motion events). The results are written to
#   <project>\data\evaluation\reference_songs\
# so you can decide which characteristics are worth training for.
#
# Nothing is trained and no audio is copied: the songs stay where they are, and
# only numbers are saved.
#
# Usage (from backend\):  python training\analyze_reference_songs.py [song name part ...]

import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import soundfile as sf

SONG_DIR = r"C:\Users\racha\Desktop\music\can be repeated"

# name part -> what the owner wants this song to teach
SONGS = [
    ("Infatuation",        "instrument separation and spatial arrangement"),
    ("Achcham Telugandham", "instrument separation and spatial arrangement"),
    ("Ordinary Person",    "bass"),
    ("Unakku Thaan",       "vocals"),
    ("Dheema.mp3",         "vocals"),
    ("Hello Rammante",     "motion of instruments in the first 30 seconds"),
    ("Insecurities",       "flute, veena, violin"),
    ("Kanave-Kanave",      "flute, veena, violin"),
    ("Peydhavi Chivarakey", "flute, veena, violin"),
    ("3 bgm",              "flute, veena, violin"),
    ("Sajra savera",       "balance of all stems"),
]
OUT_DIR = os.path.join(paths.EVAL_DIR, "reference_songs")
BANDS = [("sub 20-60 Hz", 20, 60), ("bass 60-250 Hz", 60, 250), ("low-mid 250-2k", 250, 2000),
         ("high-mid 2-8k", 2000, 8000), ("air 8k+", 8000, 20000)]


def find_song(part):
    for f in sorted(os.listdir(SONG_DIR)):
        if part.lower() in f.lower() and f.lower().endswith((".mp3", ".m4a", ".wav", ".flac")):
            return os.path.join(SONG_DIR, f)
    return None


def ffmpeg_exe():
    import static_ffmpeg
    base = os.path.join(os.path.dirname(static_ffmpeg.__file__), "bin", "win32", "ffmpeg.exe")
    return base if os.path.exists(base) else "ffmpeg"


def decode(path, out_wav):
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", path, "-ac", "2", "-ar", "44100",
                    "-c:a", "pcm_f32le", out_wav], check=True)


def stereo_stats(stereo, sr):
    """Stereo image of the ORIGINAL mix: width and left/right correlation per band."""
    left, right = stereo[:, 0].astype(np.float64), stereo[:, 1].astype(np.float64)
    n = min(len(left), sr * 240)                       # first 4 minutes is plenty
    window = np.hanning(8192)
    mid_e = side_e = ll = rr = lr = None
    spec_l = spec_r = 0
    freqs = np.fft.rfftfreq(8192, 1.0 / sr)
    acc_m = np.zeros(len(freqs)); acc_s = np.zeros(len(freqs))
    acc_ll = np.zeros(len(freqs)); acc_rr = np.zeros(len(freqs)); acc_lr = np.zeros(len(freqs), dtype=complex)
    for start in range(0, n - 8192, 8192 * 4):
        fl = np.fft.rfft(left[start:start + 8192] * window)
        fr = np.fft.rfft(right[start:start + 8192] * window)
        m, s = (fl + fr) / 2, (fl - fr) / 2
        acc_m += np.abs(m) ** 2; acc_s += np.abs(s) ** 2
        acc_ll += np.abs(fl) ** 2; acc_rr += np.abs(fr) ** 2; acc_lr += fl * np.conj(fr)
    out = {}
    total_e = acc_ll.sum() + acc_rr.sum() + 1e-12
    for name, lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        m, s = acc_m[sel].sum(), acc_s[sel].sum()
        corr = float(np.abs(acc_lr[sel].sum()) / (np.sqrt(acc_ll[sel].sum() * acc_rr[sel].sum()) + 1e-12))
        balance = 10 * np.log10((acc_ll[sel].sum() + 1e-12) / (acc_rr[sel].sum() + 1e-12))
        out[name] = {"energy_share_pct": round(100 * float((acc_ll[sel].sum() + acc_rr[sel].sum()) / total_e), 1),
                     "side_to_mid_db": round(float(10 * np.log10((s + 1e-12) / (m + 1e-12))), 1),
                     "lr_correlation": round(corr, 2),
                     "left_minus_right_db": round(float(balance), 1)}
    return out


def main():
    wanted = [a.lower() for a in sys.argv[1:]]
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(os.environ.get("TEMP", "."), "raga_ref")
    os.makedirs(tmp, exist_ok=True)
    from core.renderer import run_full_pipeline

    for part, purpose in SONGS:
        if wanted and not any(w in part.lower() for w in wanted):
            continue
        path = find_song(part)
        key = "".join(c if c.isalnum() else "_" for c in part)[:30]
        out_json = os.path.join(OUT_DIR, key + ".json")
        if path is None:
            print("NOT FOUND: " + part, flush=True)
            continue
        if os.path.exists(out_json):
            print("already analysed: " + part, flush=True)
            continue
        t0 = time.time()
        print("analysing: " + os.path.basename(path), flush=True)
        wav = os.path.join(tmp, key + ".wav")
        decode(path, wav)
        stereo, sr = sf.read(wav, dtype="float32", always_2d=True)
        result = {"song": os.path.basename(path), "purpose": purpose,
                  "duration_sec": round(len(stereo) / sr, 1),
                  "stereo_by_band": stereo_stats(stereo, sr)}

        job = "ref_" + key
        out = run_full_pipeline(wav, job)
        result["tempo_bpm"] = out["tempo_bpm"]
        result["stems"] = [{
            "stem": s["name"], "active": s["active"], "share_pct": s["share_pct"],
            "active_time_pct": s["active_pct"], "instruments": s["instruments"],
            "motion_events": s["motion_count"]} for s in out["stems8"]]
        result["motion_events"] = out["motion_events"]
        result["motion_events_first_30s"] = [e for e in out["motion_events"] if e["start_sec"] < 30]
        with open(out_json, "w") as f:
            json.dump(result, f, indent=1)

        # keep numbers only: remove the audio the pipeline wrote for this job
        for sub in ("stems", "spatial", "json"):
            d = os.path.join(paths.OUTPUTS_DIR, sub)
            for name in os.listdir(d):
                if name.startswith(job + "_"):
                    os.remove(os.path.join(d, name))
        os.remove(wav)
        print("  done in %.0f s" % (time.time() - t0), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
