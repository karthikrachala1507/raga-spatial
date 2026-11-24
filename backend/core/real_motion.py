"""
Raga Spatial - real_motion.py
Motion that follows what the ORIGINAL stereo recording really does.

The rule-based detector in motion.py guesses movement from the tune (rising or
falling phrases). Measured against the originals it did not line up with any real
panning (data/evaluation/reference_songs/MOTION_VS_REAL.md). This module measures
the movement instead:

  1. The separator's masks do not depend on the signal they are applied to, so the
     same masks are applied to the left and the right channel of the original.
  2. For each stem, the left/right balance (pan, -1..+1) is measured every 0.5 s.
  3. A stem counts as moving only if its own pan clearly changes (spread >= 0.15 and
     a swing > 0.25 inside 4 s for >= 25% of the time it is audible). In a synthetic
     test, stems held still stay under 0.12 and swept stems reach 0.25-0.55
     (SYNTHETIC_MOTION.md).
  4. The measured pan, relative to the stem's usual position, is turned into an
     azimuth offset around the stem's planned place, so stems never leave their zone
     by more than +-AZ_MAX degrees.

Vocals and bass are never moved (they stay anchored front and centre).
"""
import numpy as np
import librosa

HOP_SEC       = 1.0
SPREAD_MIN    = 0.20
SWING         = 0.25
MOVING_FRAC   = 0.50
MIN_SHARE     = 0.05
AZ_PER_PAN    = 60.0
AZ_MAX        = 70.0
SMOOTH_SEC    = 2.0
MIN_EVENT_SEC = 2.0
NEVER_MOVE    = ("vocals", "bass")


def load_stereo(path, sr):
    x, _ = librosa.load(path, sr=sr, mono=False)
    if x.ndim == 1:
        x = np.stack([x, x])
    return x[:2].astype(np.float32)


def _pan_series(l, r, sr):
    hop = int(HOP_SEC * sr)
    n = len(l) // hop
    if n < 4:
        return np.zeros(0), np.zeros(0)
    pl = (l[:n * hop].reshape(n, hop) ** 2).sum(axis=1)
    pr = (r[:n * hop].reshape(n, hop) ** 2).sum(axis=1)
    return (pr - pl) / (pr + pl + 1e-12), pr + pl


def analyse_stem(stem_l, stem_r, sr):
    """Returns dict(moving, spread, frac, pan_dev (per HOP_SEC), loud) or None."""
    pan, en = _pan_series(stem_l, stem_r, sr)
    if len(pan) < 8:
        return None
    loud = en > 0.05 * np.median(en + 1e-12)
    if loud.sum() < 8:
        return None
    k = int(round(4.0 / HOP_SEC))
    idx = [j for j in range(len(pan) - k) if loud[j:j + k].all()]
    if not idx:
        return None
    sw = np.array([pan[j:j + k].max() - pan[j:j + k].min() for j in idx])
    frac = float((sw > SWING).mean())
    spread = float(pan[loud].std())
    centre = float(np.median(pan[loud]))
    dev = np.where(loud, pan - centre, 0.0)
    ker = np.ones(max(1, int(SMOOTH_SEC / HOP_SEC))) / max(1, int(SMOOTH_SEC / HOP_SEC))
    dev = np.convolve(dev, ker, mode="same")
    swing_t = np.zeros(len(pan), bool)
    for j, s in zip(idx, sw):
        if s > SWING:
            swing_t[j:j + k] = True
    return {"moving": spread >= SPREAD_MIN and frac >= MOVING_FRAC,
            "spread": spread, "frac": frac, "dev": dev, "swing_t": swing_t}


def _events(swing_t, stem, label):
    evs, i, n = [], 0, len(swing_t)
    while i < n:
        if swing_t[i]:
            j = i
            while j < n and swing_t[j]:
                j += 1
            s, e = i * HOP_SEC, j * HOP_SEC
            if e - s >= MIN_EVENT_SEC:
                evs.append({"start_sec": s, "end_sec": e, "duration_sec": e - s,
                            "type": "panning", "spatial_motion": "original_pan",
                            "dominant_instrument": label, "confidence": 1.0,
                            "pitch_sequence": [], "stem": stem})
            i = j
        else:
            i += 1
    return evs


def detect_real_motion(audio_path, sr, separation, duration_sec, plan, levels, labels,
                       trajectory_res_sec):
    """
    separation : dict from separator_8stem.separate_8stems (needs 'estimates_16k', 'names')
    plan       : {stem: {'azimuth', 'distance'}}   levels: {stem: {'active', 'share'}}
    labels     : {stem: display label}
    Returns (events, trajectories, moving, report) - trajectories/moving keyed by stem.
    """
    from core import separator_8stem as sep
    est, names = separation["estimates_16k"], separation["names"]
    stereo = load_stereo(audio_path, sr)
    n = len(separation["audio"][next(iter(separation["audio"]))])
    left = np.pad(stereo[0], (0, max(0, n - stereo.shape[1])))[:n]
    right = np.pad(stereo[1], (0, max(0, n - stereo.shape[1])))[:n]
    stems_l, shares = sep.fullband_stems(left, est, sr)
    stems_r, _ = sep.fullband_stems(right, est, sr)

    grid_n = int(np.ceil(duration_sec / trajectory_res_sec)) + 1
    t_grid = np.arange(grid_n) * trajectory_res_sec
    events, trajectories, moving, report = [], {}, {}, {}
    for i, name in enumerate(names):
        if name in NEVER_MOVE or name not in plan:
            continue
        if not levels.get(name, {}).get("active") or shares[i] < MIN_SHARE:
            continue
        res = analyse_stem(stems_l[i], stems_r[i], sr)
        if res is None:
            continue
        report[name] = {"spread": round(res["spread"], 2), "moving_frac": round(res["frac"], 2),
                        "moving": bool(res["moving"])}
        if not res["moving"]:
            continue
        base = float(plan[name]["azimuth"])
        side = 1.0 if np.cos(np.radians(base)) >= 0 else -1.0   # rear stems: right is toward 90 deg
        t_pan = (np.arange(len(res["dev"])) + 0.5) * HOP_SEC
        dev = np.interp(t_grid, t_pan, res["dev"])
        off = np.clip(dev * AZ_PER_PAN * side, -AZ_MAX, AZ_MAX)
        trajectories[name] = ((base + off) % 360.0).astype(np.float32)
        moving[name] = np.interp(t_grid, t_pan, res["swing_t"].astype(float)) > 0.5
        events.extend(_events(res["swing_t"], name, labels.get(name, name)))
    events.sort(key=lambda e: e["start_sec"])
    return events, trajectories, moving, report
