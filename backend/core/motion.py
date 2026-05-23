# Raga Spatial - core/motion.py
# Phase 6: Motion Event Detection
# Uses relative activity scoring — motion = above-average melodic activity
# Tuned for cinematic South Indian / Harris Jayaraj style

import numpy as np
from scipy.ndimage import uniform_filter1d

NOTE_NAMES        = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHROMA_HOP_SEC    = 512 / 44100   # ~0.0116s per frame

ACTIVITY_WINDOW_SEC  = 2.0    # analyse in 2s chunks
ACTIVITY_HOP_SEC     = 0.5    # step every 0.5s
MIN_MOTION_SEC       = 1.5    # minimum event duration in seconds
MERGE_GAP_SEC        = 1.5    # merge events with gap smaller than this
MIN_ONSET_DENSITY    = 1.5    # absolute minimum onsets/sec to count as active

# Relative threshold — windows must be this far ABOVE the song's mean score
# 0.0 = everything above mean, 0.15 = only clearly above-average sections
RELATIVE_THRESHOLD   = 0.03

MELODIC_PRIORITY = [
    "Sitar","Veena","Plucked string instrument","Guitar",
    "Electric guitar","Acoustic guitar","Steel guitar",
    "Violin fiddle","String section","Bowed string instrument","Cello",
    "Flute","Saxophone","Clarinet","Wind instrument",
    "Piano","Electric piano","Keyboard musical","Synthesizer",
    "Tabla","Percussion","Drum kit","Music",
]


def detect_motion_events(chroma, onsets, beats, detections,
                          sr=44100, hop_length=512):
    chroma_hop = hop_length / sr
    total_sec  = chroma.shape[1] * chroma_hop

    print("[Motion] Starting motion detection")
    print("[Motion] Chroma: " + str(chroma.shape)
          + " | Onsets: " + str(len(onsets))
          + " | Beats: "  + str(len(beats)))

    # Smooth chroma
    chroma_smooth  = uniform_filter1d(chroma.astype(np.float32), size=5, axis=1)
    dominant_pitch = np.argmax(chroma_smooth, axis=0)

    # Compute activity scores for every window
    activity_scores = _compute_activity_scores(
        dominant_pitch, onsets, chroma_hop, total_sec
    )

    if not activity_scores:
        print("[Motion] No activity windows computed")
        return []

    # Print score distribution for debugging
    scores     = [w["score"] for w in activity_scores]
    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    threshold  = mean_score + RELATIVE_THRESHOLD

    print("[Motion] Score stats: mean=" + str(round(mean_score,3))
          + " std=" + str(round(std_score,3))
          + " threshold=" + str(round(threshold,3)))
    print("[Motion] Windows above threshold: "
          + str(sum(1 for s in scores if s >= threshold))
          + " / " + str(len(scores)))

    # Convert to segments using relative threshold
    raw_segments = _scores_to_segments(activity_scores, threshold)

    # Enrich with instrument and direction
    motion_events = _enrich_segments(
        raw_segments, dominant_pitch, chroma_hop, onsets, detections
    )

    # Merge and filter
    motion_events = _merge_and_filter(motion_events)

    print("[Motion] Motion events found: " + str(len(motion_events)))
    for ev in motion_events:
        print("[Motion]   " + str(ev["start_sec"]) + "s-" + str(ev["end_sec"]) + "s"
              + " | " + ev["type"]
              + " | " + ev["dominant_instrument"]
              + " | conf=" + str(round(ev["confidence"], 2))
              + " | " + ev["spatial_motion"])

    return motion_events


def _compute_activity_scores(dominant_pitch, onsets, chroma_hop, total_sec):
    N       = len(dominant_pitch)
    results = []
    t       = 0.0

    while t < total_sec:
        t_end   = min(t + ACTIVITY_WINDOW_SEC, total_sec)
        f_start = int(t     / chroma_hop)
        f_end   = min(int(t_end / chroma_hop), N)

        if f_end <= f_start + 2:
            t += ACTIVITY_HOP_SEC
            continue

        window_pitch = dominant_pitch[f_start:f_end]

        # Pitch change rate
        pitch_changes  = int(np.sum(np.diff(window_pitch) != 0))
        change_rate    = pitch_changes / max(len(window_pitch) - 1, 1)

        # Unique pitch classes (variety)
        unique_pitches = len(set(window_pitch.tolist()))

        # Onset density
        mask          = (onsets >= t) & (onsets < t_end)
        onset_count   = int(np.sum(mask))
        dur           = t_end - t
        onset_density = onset_count / dur if dur > 0 else 0.0

        # Direction analysis
        diffs      = np.diff(window_pitch.astype(np.int32))
        circ_diffs = np.where(diffs > 6, diffs - 12,
                     np.where(diffs < -6, diffs + 12, diffs))
        ascending  = int(np.sum(circ_diffs > 0))
        descending = int(np.sum(circ_diffs < 0))
        total_dir  = ascending + descending
        bias       = (ascending - descending) / total_dir if total_dir > 0 else 0.0

        # Composite score (all normalised 0-1)
        change_score  = min(change_rate   / 0.5,  1.0)   # 50% change rate = max
        variety_score = min(unique_pitches / 7.0,  1.0)   # 7 unique = max
        onset_score   = min(onset_density  / 6.0,  1.0)   # 6/s = max

        score = (change_score  * 0.45 +
                 variety_score * 0.30 +
                 onset_score   * 0.25)

        # Hard filter: if onset density is too low, not a real melody
        if onset_density < MIN_ONSET_DENSITY:
            score = 0.0

        results.append({
            "start_sec":     round(t,     3),
            "end_sec":       round(t_end, 3),
            "score":         round(score, 4),
            "pitch_changes": pitch_changes,
            "unique_pitches":unique_pitches,
            "onset_density": round(onset_density, 2),
            "ascending":     ascending,
            "descending":    descending,
            "direction_bias":round(bias, 3),
        })

        t += ACTIVITY_HOP_SEC

    return results


def _scores_to_segments(activity_scores, threshold):
    """Convert score series to segments using adaptive threshold."""
    segments  = []
    in_seg    = False
    seg_start = None
    seg_wins  = []

    for w in activity_scores:
        if w["score"] >= threshold:
            if not in_seg:
                in_seg    = True
                seg_start = w["start_sec"]
                seg_wins  = []
            seg_wins.append(w)
        else:
            if in_seg:
                _close_segment(seg_start, seg_wins, segments)
                in_seg   = False
                seg_wins = []

    if in_seg and seg_wins:
        _close_segment(seg_start, seg_wins, segments)

    return segments


def _close_segment(seg_start, seg_wins, segments):
    seg_end = seg_wins[-1]["end_sec"]
    dur     = seg_end - seg_start
    if dur >= MIN_MOTION_SEC:
        segments.append({
            "start_sec":   seg_start,
            "end_sec":     seg_end,
            "duration_sec":round(dur, 3),
            "windows":     seg_wins,
            "avg_score":   round(float(np.mean([w["score"] for w in seg_wins])), 4),
            "peak_score":  round(float(np.max( [w["score"] for w in seg_wins])), 4),
        })


def _enrich_segments(segments, dominant_pitch, chroma_hop, onsets, detections):
    result = []
    for seg in segments:
        windows    = seg["windows"]
        total_asc  = sum(w["ascending"]  for w in windows)
        total_desc = sum(w["descending"] for w in windows)
        total_dir  = total_asc + total_desc

        asc_ratio  = total_asc / total_dir if total_dir > 0 else 0.5

        if asc_ratio > 0.57:
            motion_type = "ascending"
            spatial     = "left_to_right"
        elif asc_ratio < 0.43:
            motion_type = "descending"
            spatial     = "right_to_left"
        else:
            motion_type = "oscillating"
            spatial     = "circular"

        # Pitch sequence sample
        f_start   = int(seg["start_sec"] / chroma_hop)
        f_end     = min(int(seg["end_sec"] / chroma_hop), len(dominant_pitch))
        step      = max(1, (f_end - f_start) // 16)
        pitch_seq = [NOTE_NAMES[int(dominant_pitch[i])]
                     for i in range(f_start, f_end, step)]

        # Onset stats
        mask        = (onsets >= seg["start_sec"]) & (onsets <= seg["end_sec"])
        onset_count = int(np.sum(mask))
        onset_dens  = round(onset_count / seg["duration_sec"], 2)

        # Dominant instrument
        dominant_instr = _find_dominant_instrument(
            seg["start_sec"], seg["end_sec"], detections
        )

        # Variety bonus
        variety_bonus = min(len(set(pitch_seq)) / 6.0, 1.0) * 0.08
        confidence    = round(min(seg["avg_score"] + variety_bonus, 1.0), 3)

        result.append({
            "start_sec":           seg["start_sec"],
            "end_sec":             seg["end_sec"],
            "duration_sec":        seg["duration_sec"],
            "type":                motion_type,
            "spatial_motion":      spatial,
            "dominant_instrument": dominant_instr,
            "pitch_sequence":      pitch_seq,
            "confidence":          confidence,
            "onset_density":       onset_dens,
            "onset_count":         onset_count,
            "ascending_moves":     total_asc,
            "descending_moves":    total_desc,
        })

    return result


def _find_dominant_instrument(start_sec, end_sec, detections):
    scores = {}
    for dw in detections:
        if dw["end_sec"] < start_sec or dw["start_sec"] > end_sec:
            continue
        for det in dw["detections"]:
            label = det["label"]
            scores[label] = max(scores.get(label, 0.0), det["confidence"])
    for candidate in MELODIC_PRIORITY:
        if candidate in scores:
            return candidate
    return "Music"


def _merge_and_filter(motion_events):
    if not motion_events:
        return []

    events = sorted(motion_events, key=lambda x: x["start_sec"])
    merged = [events[0]]

    for ev in events[1:]:
        last = merged[-1]
        gap  = ev["start_sec"] - last["end_sec"]
        if gap <= MERGE_GAP_SEC:
            last["end_sec"]           = max(last["end_sec"], ev["end_sec"])
            last["duration_sec"]      = round(last["end_sec"] - last["start_sec"], 3)
            last["confidence"]        = round(max(last["confidence"], ev["confidence"]), 3)
            last["pitch_sequence"]   += ev["pitch_sequence"]
            last["onset_count"]      += ev["onset_count"]
            last["ascending_moves"]  += ev["ascending_moves"]
            last["descending_moves"] += ev["descending_moves"]
            total = last["ascending_moves"] + last["descending_moves"]
            if total > 0:
                r = last["ascending_moves"] / total
                if r > 0.57:
                    last["type"] = "ascending"; last["spatial_motion"] = "left_to_right"
                elif r < 0.43:
                    last["type"] = "descending"; last["spatial_motion"] = "right_to_left"
                else:
                    last["type"] = "oscillating"; last["spatial_motion"] = "circular"
        else:
            merged.append(ev)

    return merged


def build_motion_timeline(motion_events, total_duration_sec):
    timeline = []
    for t in np.arange(0.0, total_duration_sec, 1.0):
        entry = {
            "time_sec":            round(float(t), 2),
            "in_motion":           False,
            "spatial_motion":      None,
            "dominant_instrument": None,
            "confidence":          0.0,
        }
        for ev in motion_events:
            if ev["start_sec"] <= t <= ev["end_sec"]:
                entry["in_motion"]           = True
                entry["spatial_motion"]      = ev["spatial_motion"]
                entry["dominant_instrument"] = ev["dominant_instrument"]
                entry["confidence"]          = ev["confidence"]
                break
        timeline.append(entry)
    return timeline


def summarise_motion(motion_events, total_duration_sec):
    if not motion_events:
        return {"total_events": 0, "motion_coverage_pct": 0.0,
                "dominant_type": "none", "dominant_instrument": "unknown"}
    total_sec = sum(ev["duration_sec"] for ev in motion_events)
    coverage  = round(total_sec / total_duration_sec * 100, 1)
    types     = [ev["type"]                for ev in motion_events]
    instrs    = [ev["dominant_instrument"] for ev in motion_events]
    return {
        "total_events":        len(motion_events),
        "motion_coverage_pct": coverage,
        "total_motion_sec":    round(total_sec, 1),
        "dominant_type":       max(set(types),  key=types.count),
        "dominant_instrument": max(set(instrs), key=instrs.count),
        "ascending_count":     types.count("ascending"),
        "descending_count":    types.count("descending"),
        "oscillating_count":   types.count("oscillating"),
    }
