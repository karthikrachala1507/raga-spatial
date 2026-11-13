# Raga Spatial - tests/test_8stem.py
# Checks for the 8-stem pipeline.
#
#   python tests\test_8stem.py            (quick audio checks, about 1 minute)
#   python tests\test_8stem.py pipeline   (whole pipeline on test.flac, a few minutes)
#
# "pipeline" uses the trained model if there is one; otherwise it makes an
# untrained one, which still checks that every stage runs and fits together.

import os
import sys
import time
import glob
import tempfile
import numpy as np

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND)
os.chdir(_BACKEND)

SR = 44100


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))
    return ok


def ripple(binaural):
    n = binaural.shape[1] // 441 * 441
    env = np.sqrt(np.mean((binaural[0] ** 2 + binaural[1] ** 2)[:n].reshape(-1, 441), axis=1))[20:-20]
    return float(np.std(np.diff(env)) / np.mean(env))


def units():
    from core import hrtf, separator8
    sofa = hrtf.load_sofa()
    results = []

    t = np.arange(SR * 2) / SR
    tone = (0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    right = hrtf.spatialise_static(tone, 90, sofa, SR)
    left = hrtf.spatialise_static(tone, 270, sofa, SR)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    results.append(check("source on the right is louder in the right ear",
                         rms(right[1]) > 1.2 * rms(right[0])))
    results.append(check("source on the left is louder in the left ear",
                         rms(left[0]) > 1.2 * rms(left[1])))

    noise = (np.random.RandomState(0).randn(SR * 6) * 0.2).astype(np.float32)
    sweep = np.linspace(270, 450, int(6 / 0.05)) % 360
    moving = hrtf.spatialise_trajectory(noise, sweep, 0.05, sofa, SR)
    still = hrtf.spatialise_static(noise, 0, sofa, SR)
    results.append(check("moving source has no tremolo", ripple(moving) < 1.3 * ripple(still),
                         "ripple %.3f vs %.3f static" % (ripple(moving), ripple(still))))
    results.append(check("sweep goes from left to right",
                         rms(moving[0, :SR]) > rms(moving[1, :SR]) and rms(moving[1, -SR:]) > rms(moving[0, -SR:])))

    near, far = hrtf.distance_params(0.2), hrtf.distance_params(0.85)
    results.append(check("farther is quieter, duller and more reverberant",
                         far["gain"] < near["gain"] and far["cutoff_hz"] < near["cutoff_hz"]
                         and far["reverb_mix"] > near["reverb_mix"]))

    x = (np.random.RandomState(1).randn(SR * 12) * 0.1).astype(np.float32)
    est = np.random.RandomState(2).randn(4, 16000 * 12).astype(np.float32) \
        * np.array([1.0, 0.5, 0.2, 0.001])[:, None]
    stems, shares = separator8.fullband_stems(x, est, SR)
    err = float(np.abs(stems.sum(0) - x)[4096:-4096].max())
    results.append(check("full-band stems add back up to the input", err < 1e-4, "max error %.1e" % err))
    results.append(check("a near-silent stem is treated as absent", shares[3] == 0.0))
    return all(results)


def pipeline():
    import torch
    import soundfile as sf
    from core import separator8
    results = []

    if not separator8.is_available():
        from train_separator import BandSplitRNN, FEATURE_DIM, N_BLOCKS
        path = os.path.join(tempfile.gettempdir(), "raga_untrained_sep8.pt")
        torch.manual_seed(0)
        torch.save({"epoch": -1,
                    "model": BandSplitRNN(n_stems=8, feature_dim=FEATURE_DIM, n_blocks=N_BLOCKS).state_dict(),
                    "config": {"feature_dim": FEATURE_DIM, "n_blocks": N_BLOCKS}}, path)
        os.environ["RAGA_SEP8_MODEL"] = path
        print("No trained model yet: using an untrained one (plumbing check only)")
    os.environ.setdefault("RAGA_SEP8_MODE", "pure")

    from core.renderer import run_full_pipeline
    from core.preprocess import load_audio, normalize_audio
    song = os.path.join(_BACKEND, "test.flac")
    for old in glob.glob(os.path.join(_BACKEND, "outputs", "*", "t8test_*")):
        os.remove(old)                      # leftovers from an earlier test run
    t = time.time()
    out = run_full_pipeline(song, "t8test")
    print("\npipeline finished in %.0f s (model epoch %s, %s mode)" % (
        time.time() - t, out["model_epoch"], out["separation_mode"]))

    mix, _ = sf.read(out["spatial_wav"])
    results.append(check("binaural mix written, stereo, not clipped",
                         mix.ndim == 2 and mix.shape[1] == 2 and np.abs(mix).max() <= 0.96))
    total = None
    for path in out["stems"].values():
        a, fs = sf.read(path)
        total = a if total is None else total + a
    ref, _ = load_audio(song)
    ref = normalize_audio(ref)
    n = min(len(ref), len(total))
    err = float(np.abs(total[:n] - ref[:n])[4096:-4096].max())
    results.append(check("the 8 stems add back up to the song", err < 2e-3, "max error %.4f" % err))
    results.append(check("stems are full bandwidth (44.1 kHz)", fs == 44100))
    results.append(check("8 stems described, layers 1 to 8",
                         [s["layer"] for s in out["stems8"]] == list(range(1, 9))))
    mp3s = glob.glob(os.path.join(_BACKEND, "outputs", "spatial", "t8test_*_binaural.mp3"))
    active = [s for s in out["stems8"] if s["active"]]
    results.append(check("one binaural file per active stem", len(mp3s) == len(active),
                         "%d files, %d active stems" % (len(mp3s), len(active))))
    results.append(check("motion coverage is a valid percentage",
                         0 <= out["motion_summary"]["motion_coverage_pct"] <= 100))
    for s in out["stems8"]:
        print("  layer %d %-16s %5.1f%% of energy  %s" % (
            s["layer"], s["label"], s["share_pct"], ", ".join(i["label"] for i in s["instruments"])))
    print("\nOpen http://localhost:3000/?job=t8test in the app to see and hear this result.")
    return all(results)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "units"
    ok = pipeline() if mode == "pipeline" else units()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
