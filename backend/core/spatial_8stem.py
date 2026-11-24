# Raga Spatial - core/spatial_8stem.py
# 8-stem spatial layout, depth layering, per-stem analysis and binaural rendering.
#
# Azimuth: 0 = front, 90 = right, 180 = rear, 270 = left.
# Distance: 0 = nearest, 1 = farthest. Layer 1 is closest to the listener.

import os
import numpy as np
import soundfile as sf

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPATIAL_DIR = os.path.join(BACKEND_DIR, "outputs", "spatial")

# trim_db compensates the distance gain where the design would otherwise thin
# the mix: bass sits far back but must keep its weight.
STEM_LAYOUT = {
    "vocals":          {"label": "Vocals",          "layer": 1, "azimuth": 0,   "distance": 0.20,
                        "trim_db": 0.0, "color": "#60A5FA",
                        "contains": "voice, humming, chorus"},
    "plucked_strings": {"label": "Plucked strings", "layer": 2, "azimuth": 315, "distance": 0.35,
                        "trim_db": 0.0, "color": "#F59E0B",
                        "contains": "veena, sitar, guitar, mandolin, santoor"},
    "wind":            {"label": "Wind",            "layer": 3, "azimuth": 45,  "distance": 0.45,
                        "trim_db": 0.0, "color": "#86EFAC",
                        "contains": "bansuri, nadaswaram, shehnai, flute, sax, brass"},
    "keys_synth":      {"label": "Keys / Synth",    "layer": 4, "azimuth": 30,  "distance": 0.50,
                        "trim_db": 0.0, "color": "#FCA5A5",
                        "contains": "piano, harmonium, synth, tanpura drone"},
    "bowed_strings":   {"label": "Bowed strings",   "layer": 5, "azimuth": 225, "distance": 0.60,
                        "trim_db": 1.0, "color": "#34D399",
                        "contains": "violin, viola, cello, string section"},
    "hand_percussion": {"label": "Hand percussion", "layer": 6, "azimuth": 120, "distance": 0.55,
                        "trim_db": 1.5, "color": "#FB7185",
                        "contains": "tabla, mridangam, ghatam, kanjira, dholak, thavil"},
    "percussion":      {"label": "Drums",           "layer": 7, "azimuth": 180, "distance": 0.65,
                        "trim_db": 2.0, "color": "#F87171",
                        "contains": "drum kit, electronic drums, cymbals"},
    "bass":            {"label": "Bass",            "layer": 8, "azimuth": 135, "distance": 0.75,
                        "trim_db": 4.0, "color": "#A78BFA",
                        "contains": "bass guitar, 808, sub"},
    "folk_texture":    {"label": "Folk / Texture",  "layer": 9, "azimuth": 270, "distance": 0.85,
                        "trim_db": 2.0, "color": "#FB923C",
                        "contains": "dappu, parai, dhol, kombu, atmosphere"},
}
STEM_ORDER = sorted(STEM_LAYOUT, key=lambda s: STEM_LAYOUT[s]["layer"])

# Stems that may sweep around the listener during melodic runs
MOTION_CARRIERS = ["plucked_strings", "wind"]

# AudioSet label families used to name what is inside each stem (BEATs)
FAMILY_WORDS = {
    "vocals":          ["singing", "vocal", "choir", "humming", "chant", "a capella", "yodel"],
    "plucked_strings": ["plucked", "guitar", "strum", "banjo", "sitar", "mandolin",
                        "ukulele", "pizzicato", "harp", "zither", "veena"],
    "wind":            ["flute", "wind instrument", "saxophone", "clarinet", "harmonica",
                        "bagpipe", "didgeridoo", "brass", "trumpet", "trombone",
                        "french horn", "oboe", "bassoon", "shofar"],
    "keys_synth":      ["piano", "keyboard", "organ", "harpsichord", "synthesizer",
                        "sampler", "accordion", "harmonium"],
    "bowed_strings":   ["bowed", "string section", "violin", "fiddle", "cello",
                        "double bass", "orchestra"],
    "percussion":      ["drum", "percussion", "snare", "hi-hat",
                        "cymbal", "gong", "marimba", "vibraphone",
                        "mallet", "steelpan", "rimshot", "timpani", "wood block",
                        "cowbell"],
    "hand_percussion": ["tabla", "mridangam", "ghatam", "hand drum", "conga", "bongo",
                        "tambourine", "maraca", "shaker", "clapping", "percussion",
                        "drum"],
    "bass":            ["bass guitar", "double bass", "bass"],
    "folk_texture":    ["drum", "percussion", "tabla", "tambourine", "ambient",
                        "folk", "bell", "chime", "rattle"],
}

ENVELOPE_HZ        = 4       # level curve sent to the UI (values per second)
TRAJECTORY_RES_SEC = 0.05
MOTION_EASE_SEC    = 0.4     # glide into and out of a motion sweep
ACTIVE_SHARE       = 0.01    # a stem below 1% of the song's energy counts as absent


# ── Analysis ──────────────────────────────────────────────────────────────────

def stem_levels(stems, sr):
    """Energy share, activity and a level envelope per stem."""
    hop   = int(sr / ENVELOPE_HZ)
    info  = {}
    total = sum(float(np.sum(a.astype(np.float64) ** 2)) for a in stems.values()) + 1e-12
    env_peak = 1e-9
    for name, audio in stems.items():
        n   = len(audio) // hop
        env = np.sqrt(np.mean(audio[:n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(1)
        env_peak = max(env_peak, float(env.max()))
        energy   = float(np.sum(audio.astype(np.float64) ** 2))
        info[name] = {"share": energy / total, "env": env,
                      "rms": float(np.sqrt(energy / max(len(audio), 1)))}
    for name, d in info.items():
        env = d["env"] / env_peak
        d["active"]     = bool(d["share"] >= ACTIVE_SHARE)
        d["active_pct"] = round(float(np.mean(env > 0.05)) * 100, 1)
        d["envelope"]   = [round(float(v), 3) for v in env]
        del d["env"]
    return info


def name_instruments(stem_name, detection_windows, top_n=3):
    """Top BEATs labels of a stem that belong to its instrument family."""
    words  = FAMILY_WORDS.get(stem_name, [])
    scores, counts = {}, {}
    for window in detection_windows:
        for det in window.get("detections", []):
            label = det["label"]
            low   = label.lower()
            if any(w in low for w in words):
                scores[label] = scores.get(label, 0.0) + det["confidence"]
                counts[label] = counts.get(label, 0) + 1
    n_windows = max(len(detection_windows), 1)
    ranked = sorted(scores, key=lambda l: scores[l] / n_windows, reverse=True)
    return [{"label": l,
             "avg_confidence": round(scores[l] / counts[l], 3),
             "presence_pct":   round(counts[l] / n_windows * 100, 1),
             "window_count":   counts[l]} for l in ranked[:top_n]]


# ── Placement planner ─────────────────────────────────────────────────────────

MIN_ANGLE_GAP  = 35.0    # degrees: two stems closer than this must differ in depth
MIN_DEPTH_GAP  = 0.20    # distance units
FRONT_RESERVED = 20.0    # +/- degrees kept for the vocals (and bass) when they play
PLAN_PRIORITY  = ["vocals", "bass", "percussion", "hand_percussion", "plucked_strings", "wind",
                  "bowed_strings", "keys_synth", "folk_texture"]


def _angle_gap(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def plan_layout(active):
    """
    Choose an azimuth and distance for each ACTIVE stem.

    A stem keeps its home position unless that collides with a stem already
    placed: closer than MIN_ANGLE_GAP degrees AND closer than MIN_DEPTH_GAP in
    depth. Stems far enough apart in either angle or depth may share a region
    of the stage. Stems that are absent leave their space free, and the front
    centre stays with the vocals while they are playing. Deterministic.
    Returns {stem: {"azimuth": deg, "distance": d}}.
    """
    placed = {}
    for name in PLAN_PRIORITY:
        if name not in active or name not in STEM_LAYOUT:
            continue
        home, dist = float(STEM_LAYOUT[name]["azimuth"]), STEM_LAYOUT[name]["distance"]

        def free(az):
            if name not in ("vocals", "bass") and "vocals" in placed                     and _angle_gap(az, 0.0) < FRONT_RESERVED:
                return False
            for other in placed.values():
                if (_angle_gap(az, other["azimuth"]) < MIN_ANGLE_GAP
                        and abs(dist - other["distance"]) < MIN_DEPTH_GAP):
                    return False
            return True

        chosen = home
        if not free(home):
            for step in range(1, 10):
                found = None
                for sign in (1, -1):
                    cand = (home + sign * 25.0 * step) % 360.0
                    if free(cand):
                        found = cand
                        break
                if found is not None:
                    chosen = found
                    break
        placed[name] = {"azimuth": round(chosen, 1), "distance": dist}
    return placed



# ── Trajectories ──────────────────────────────────────────────────────────────

def build_trajectory(stem_name, motion_events, duration_sec, base=None):
    """Azimuth of a stem over time. Returns (azimuths, in_motion flags)."""
    from core.spatial import _motion_azimuth
    base  = float(STEM_LAYOUT[stem_name]["azimuth"] if base is None else base)
    n     = int(np.ceil(duration_sec / TRAJECTORY_RES_SEC)) + 1
    az    = np.full(n, base, dtype=np.float32)
    moving = np.zeros(n, dtype=bool)
    for ev in motion_events:
        start, end = float(ev["start_sec"]), float(ev["end_sec"])
        length = max(end - start, 1e-3)
        i0 = int(start / TRAJECTORY_RES_SEC)
        i1 = min(int(end / TRAJECTORY_RES_SEC) + 1, n)
        for i in range(i0, i1):
            t        = i * TRAJECTORY_RES_SEC
            progress = min(max((t - start) / length, 0.0), 1.0)
            target   = _motion_azimuth(base, ev["spatial_motion"], progress)
            # glide from/to the home position at the edges of the event
            edge  = min(t - start, end - t)
            blend = min(max(edge / MOTION_EASE_SEC, 0.0), 1.0)
            diff  = ((target - base + 180.0) % 360.0) - 180.0
            az[i] = (base + diff * blend) % 360.0
            moving[i] = True
    return az, moving


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_stems(stems, sr, job_id, trajectories=None, levels=None, progress=None, plan=None):
    """
    Place every stem at its position and depth and mix to binaural stereo.

    stems        : {stem_name: mono array at sr}
    trajectories : {stem_name: azimuth array at TRAJECTORY_RES_SEC} for moving stems
    Writes <job>_spatial.wav (the mix) and <job>_<stem>_binaural.mp3 (one per
    stem, for the live mixer in the UI). Returns (mix_path, {stem: mp3_path}).
    """
    from core import hrtf
    os.makedirs(SPATIAL_DIR, exist_ok=True)
    trajectories = trajectories or {}
    sofa   = hrtf.load_sofa()
    n      = max(len(a) for a in stems.values())
    placed = {}

    for seed, name in enumerate(STEM_ORDER):
        if name not in stems:
            continue
        if levels is not None and not levels[name]["active"]:
            continue
        layout = STEM_LAYOUT[name]
        audio  = np.pad(stems[name].astype(np.float32), (0, n - len(stems[name])))
        plan_i = (plan or {}).get(name, {})
        dist   = plan_i.get("distance", layout["distance"])
        azim   = plan_i.get("azimuth", layout["azimuth"])
        trim   = 10.0 ** (layout["trim_db"] / 20.0)
        if progress:
            progress("Placing " + layout["label"] + " (layer " + str(layout["layer"]) + ")")

        dry = hrtf.apply_distance_effect(audio, dist, sr, lowpass=(name != "bass")) * trim
        if name in trajectories:
            binaural = hrtf.spatialise_trajectory(dry, trajectories[name],
                                                  TRAJECTORY_RES_SEC, sofa, sr)
        else:
            binaural = hrtf.spatialise_static(dry, azim, sofa, sr)
        if name != "bass":                    # reverb on bass only muddies it
            binaural = binaural + hrtf.reverb_tail(audio, dist, sr, seed=seed + 1) * trim
        placed[name] = binaural

    if not placed:
        raise RuntimeError("No active stems to render")

    mix   = np.sum(list(placed.values()), axis=0)
    scale = 0.95 / max(float(np.abs(mix).max()), 1e-8)     # same scale for mix and stems
    mix_path = os.path.join(SPATIAL_DIR, job_id + "_spatial.wav")
    sf.write(mix_path, (mix * scale).T, sr, subtype="PCM_16")

    stem_paths = {}
    for name, binaural in placed.items():
        path = os.path.join(SPATIAL_DIR, job_id + "_" + name + "_binaural.mp3")
        sf.write(path, np.clip(binaural * scale, -1.0, 1.0).T, sr, format="MP3")
        stem_paths[name] = path
    print("[Spatial8] Rendered " + str(len(placed)) + " stems -> " + mix_path)
    return mix_path, stem_paths
