# Raga Spatial - core/spatial.py
# Phase 7: Spatial Position Assignment
#
# Assigns each detected instrument a fixed 3D position around the listener.
# During motion events, the dominant instrument moves along a trajectory.
# All positions expressed as azimuth (0-360 degrees) and distance (0.0-1.0).
#
# Azimuth convention:
#   0   = front center
#   90  = right
#   180 = rear center
#   270 = left
#   (clockwise when viewed from above)

import numpy as np
import json
import os

# ── Fixed spatial positions for each instrument class ────────────────────────
# Based on project report + standard cinematic mixing conventions
# azimuth: 0=front, 90=right, 180=rear, 270=left
# distance: 0.0=very close, 1.0=very far (0.3-0.6 = natural cinematic depth)

INSTRUMENT_POSITIONS = {
    # Vocals always center front — human ear anchors to voice
    "Singing":                   {"azimuth": 0,   "distance": 0.25, "label": "Vocals"},
    "Vocal music":               {"azimuth": 0,   "distance": 0.25, "label": "Vocals"},
    "Child singing":             {"azimuth": 0,   "distance": 0.25, "label": "Vocals"},
    "Choir":                     {"azimuth": 355, "distance": 0.35, "label": "Choir"},
    "Humming":                   {"azimuth": 5,   "distance": 0.3,  "label": "Vocals"},
    "Chant":                     {"azimuth": 0,   "distance": 0.3,  "label": "Vocals"},

    # Plucked strings — front left (lead melodic position)
    "Plucked string instrument": {"azimuth": 315, "distance": 0.35, "label": "Plucked strings"},
    "Guitar":                    {"azimuth": 315, "distance": 0.35, "label": "Guitar"},
    "Electric guitar":           {"azimuth": 310, "distance": 0.4,  "label": "Electric guitar"},
    "Acoustic guitar":           {"azimuth": 320, "distance": 0.35, "label": "Acoustic guitar"},
    "Steel guitar":              {"azimuth": 310, "distance": 0.4,  "label": "Steel guitar"},
    "Strum":                     {"azimuth": 315, "distance": 0.35, "label": "Guitar strum"},
    "Tapping guitar technique":  {"azimuth": 315, "distance": 0.35, "label": "Guitar tap"},
    "Banjo":                     {"azimuth": 320, "distance": 0.4,  "label": "Banjo"},
    "Sitar":                     {"azimuth": 315, "distance": 0.35, "label": "Sitar"},
    "Veena":                     {"azimuth": 315, "distance": 0.35, "label": "Veena"},
    "Mandolin":                  {"azimuth": 325, "distance": 0.4,  "label": "Mandolin"},
    "Ukulele":                   {"azimuth": 320, "distance": 0.4,  "label": "Ukulele"},
    "Pizzicato":                 {"azimuth": 315, "distance": 0.4,  "label": "Pizzicato"},

    # Flute / Wind — front right
    "Flute":                     {"azimuth": 45,  "distance": 0.4,  "label": "Flute"},
    "Wind instrument":           {"azimuth": 45,  "distance": 0.4,  "label": "Wind"},
    "Saxophone":                 {"azimuth": 50,  "distance": 0.4,  "label": "Saxophone"},
    "Clarinet":                  {"azimuth": 40,  "distance": 0.4,  "label": "Clarinet"},
    "Harmonica":                 {"azimuth": 45,  "distance": 0.45, "label": "Harmonica"},
    "Accordion":                 {"azimuth": 50,  "distance": 0.45, "label": "Accordion"},
    "Bagpipes":                  {"azimuth": 55,  "distance": 0.5,  "label": "Bagpipes"},
    "Didgeridoo":                {"azimuth": 55,  "distance": 0.5,  "label": "Didgeridoo"},

    # Brass — right side
    "Brass instrument":          {"azimuth": 80,  "distance": 0.5,  "label": "Brass"},
    "Trumpet":                   {"azimuth": 75,  "distance": 0.45, "label": "Trumpet"},
    "Trombone":                  {"azimuth": 85,  "distance": 0.5,  "label": "Trombone"},
    "French horn":               {"azimuth": 80,  "distance": 0.5,  "label": "French horn"},

    # Bowed strings — rear left (orchestral string section position)
    "Bowed string instrument":   {"azimuth": 225, "distance": 0.5,  "label": "Strings"},
    "String section":            {"azimuth": 220, "distance": 0.5,  "label": "String section"},
    "Violin fiddle":             {"azimuth": 230, "distance": 0.45, "label": "Violin"},
    "Cello":                     {"azimuth": 215, "distance": 0.55, "label": "Cello"},
    "Double bass":               {"azimuth": 210, "distance": 0.6,  "label": "Double bass"},
    "Orchestra":                 {"azimuth": 180, "distance": 0.55, "label": "Orchestra"},

    # Keyboard / Piano — right of center front
    "Piano":                     {"azimuth": 30,  "distance": 0.4,  "label": "Piano"},
    "Electric piano":            {"azimuth": 30,  "distance": 0.4,  "label": "Electric piano"},
    "Keyboard musical":          {"azimuth": 35,  "distance": 0.4,  "label": "Keyboard"},
    "Organ":                     {"azimuth": 20,  "distance": 0.5,  "label": "Organ"},
    "Hammond organ":             {"azimuth": 20,  "distance": 0.5,  "label": "Hammond organ"},
    "Harpsichord":               {"azimuth": 25,  "distance": 0.5,  "label": "Harpsichord"},
    "Synthesizer":               {"azimuth": 180, "distance": 0.7,  "label": "Synth pad"},
    "Sampler":                   {"azimuth": 180, "distance": 0.7,  "label": "Sampler"},

    # Percussion / Tabla — rear center (rhythm section behind listener)
    "Tabla":                     {"azimuth": 175, "distance": 0.5,  "label": "Tabla"},
    "Mridangam":                 {"azimuth": 180, "distance": 0.5,  "label": "Mridangam"},
    "Drum kit":                  {"azimuth": 185, "distance": 0.5,  "label": "Drums"},
    "Drum machine":              {"azimuth": 185, "distance": 0.5,  "label": "Drum machine"},
    "Drum":                      {"azimuth": 180, "distance": 0.5,  "label": "Drum"},
    "Snare drum":                {"azimuth": 170, "distance": 0.45, "label": "Snare"},
    "Bass drum":                 {"azimuth": 190, "distance": 0.55, "label": "Kick drum"},
    "Percussion":                {"azimuth": 180, "distance": 0.5,  "label": "Percussion"},
    "Hi-hat":                    {"azimuth": 165, "distance": 0.45, "label": "Hi-hat"},
    "Cymbal":                    {"azimuth": 160, "distance": 0.45, "label": "Cymbal"},
    "Tambourine":                {"azimuth": 170, "distance": 0.45, "label": "Tambourine"},
    "Gong":                      {"azimuth": 175, "distance": 0.6,  "label": "Gong"},
    "Marimba xylophone":         {"azimuth": 155, "distance": 0.5,  "label": "Marimba"},
    "Vibraphone":                {"azimuth": 155, "distance": 0.5,  "label": "Vibraphone"},
    "Mallet percussion":         {"azimuth": 160, "distance": 0.5,  "label": "Mallet perc"},
    "Steelpan":                  {"azimuth": 150, "distance": 0.5,  "label": "Steelpan"},

    # Bass — rear right
    "Bass guitar":               {"azimuth": 135, "distance": 0.55, "label": "Bass guitar"},

    # Bell / Chime — front right (bright, airy)
    "Bell":                      {"azimuth": 60,  "distance": 0.5,  "label": "Bell"},
    "Chime":                     {"azimuth": 55,  "distance": 0.5,  "label": "Chime"},
    "Wind chime":                {"azimuth": 50,  "distance": 0.6,  "label": "Wind chime"},
    "Singing bowl":              {"azimuth": 60,  "distance": 0.6,  "label": "Singing bowl"},
    "Theremin":                  {"azimuth": 60,  "distance": 0.5,  "label": "Theremin"},

    # Genre/style labels — diffuse surround
    "Music":                     {"azimuth": 0,   "distance": 0.5,  "label": "Music"},
    "Musical instrument":        {"azimuth": 0,   "distance": 0.45, "label": "Instrument"},
    "Ambient music":             {"azimuth": 180, "distance": 0.8,  "label": "Ambient"},
    "New-age music":             {"azimuth": 180, "distance": 0.75, "label": "Cinematic pad"},
    "Swing music":               {"azimuth": 180, "distance": 0.7,  "label": "Cinematic"},
    "Carnatic music":            {"azimuth": 0,   "distance": 0.5,  "label": "Carnatic"},
    "Classical music":           {"azimuth": 0,   "distance": 0.5,  "label": "Classical"},
    "Soundtrack music":          {"azimuth": 180, "distance": 0.7,  "label": "Soundtrack"},
}

# Default position for any instrument not in the map
DEFAULT_POSITION = {"azimuth": 0, "distance": 0.5, "label": "Unknown"}

# ── Motion trajectory definitions ────────────────────────────────────────────
# For each spatial_motion type, define how the azimuth changes over time
# as a fraction of the event duration (0.0 = start, 1.0 = end)

def _motion_azimuth(base_azimuth, spatial_motion, progress):
    """
    Compute the azimuth of a moving instrument at a given point in time.

    Args:
        base_azimuth  : the instrument's default azimuth position
        spatial_motion: "left_to_right", "right_to_left", or "circular"
        progress      : 0.0 to 1.0 (position within the motion event)

    Returns:
        azimuth angle (0-360)
    """
    if spatial_motion == "left_to_right":
        # Sweep from 270 (left) to 90 (right) through front center
        start_az = 270
        end_az   = 90
        # Use smooth sine interpolation for natural movement
        t        = (1 - np.cos(progress * np.pi)) / 2   # ease in-out
        az       = start_az + t * ((end_az - start_az + 360) % 360)
        return az % 360

    elif spatial_motion == "right_to_left":
        # Sweep from 90 (right) to 270 (left) through front center
        start_az = 90
        end_az   = 270
        t        = (1 - np.cos(progress * np.pi)) / 2
        az       = start_az + t * ((end_az - start_az + 360) % 360)
        return az % 360

    elif spatial_motion == "circular":
        # Full circular sweep starting from base position
        # Goes front -> right -> rear -> left -> front
        az = (base_azimuth + progress * 360) % 360
        return az

    else:
        return base_azimuth


def assign_spatial_positions(detections, motion_events, total_duration_sec,
                              resolution_sec=0.1):
    """
    Main entry point for Phase 7.

    Produces a time-indexed spatial map — for every time step, the azimuth
    and distance of every active instrument.

    Args:
        detections        : merged detection results from Phase 5
        motion_events     : motion events from Phase 6
        total_duration_sec: total song duration
        resolution_sec    : time step in seconds (default 0.1s = 100ms)

    Returns:
        {
            "resolution_sec": float,
            "total_duration_sec": float,
            "instrument_positions": {label: {azimuth, distance}},
            "timeline": [
                {
                    "time_sec": float,
                    "instruments": [
                        {
                            "label": str,
                            "azimuth": float,
                            "distance": float,
                            "in_motion": bool,
                            "confidence": float,
                        }
                    ]
                },
                ...
            ]
        }
    """
    print("[Spatial] Starting spatial assignment")
    print("[Spatial] Duration: " + str(round(total_duration_sec, 1)) + "s"
          + " | Resolution: " + str(resolution_sec) + "s"
          + " | Motion events: " + str(len(motion_events)))

    # Step 1: Build instrument presence timeline from detections
    # For each time step, which instruments are active and at what confidence?
    print("[Spatial] Building instrument presence timeline...")
    presence = _build_presence_timeline(detections, total_duration_sec, resolution_sec)

    # Step 2: Build motion event lookup
    # For each time step, is there a motion event? If so, what instrument moves?
    motion_lookup = _build_motion_lookup(motion_events, total_duration_sec, resolution_sec)

    # Step 3: For each time step, compute final positions
    print("[Spatial] Computing positions...")
    timeline = []
    times    = np.arange(0.0, total_duration_sec, resolution_sec)

    for i, t in enumerate(times):
        t_round      = round(float(t), 3)
        active_instr = presence.get(i, [])
        motion_info  = motion_lookup.get(i, None)

        instruments_at_t = []
        for instr in active_instr:
            label      = instr["label"]
            confidence = instr["confidence"]
            pos        = INSTRUMENT_POSITIONS.get(label, DEFAULT_POSITION)
            azimuth    = pos["azimuth"]
            distance   = pos["distance"]
            in_motion  = False

            # If this instrument is the dominant moving instrument
            if (motion_info is not None and
                    motion_info["dominant_instrument"] == label):
                progress  = motion_info["progress"]
                azimuth   = _motion_azimuth(
                    azimuth,
                    motion_info["spatial_motion"],
                    progress
                )
                azimuth   = round(azimuth, 1)
                in_motion = True

            instruments_at_t.append({
                "label":      pos.get("label", label),
                "raw_label":  label,
                "azimuth":    round(azimuth, 1),
                "distance":   distance,
                "in_motion":  in_motion,
                "confidence": round(confidence, 3),
            })

        # Sort by confidence descending
        instruments_at_t.sort(key=lambda x: x["confidence"], reverse=True)

        timeline.append({
            "time_sec":    t_round,
            "instruments": instruments_at_t,
        })

    # Step 4: Build static position map (for frontend spatial map display)
    static_positions = {}
    for label, pos in INSTRUMENT_POSITIONS.items():
        static_positions[pos["label"]] = {
            "azimuth":  pos["azimuth"],
            "distance": pos["distance"],
        }

    result = {
        "resolution_sec":      resolution_sec,
        "total_duration_sec":  round(total_duration_sec, 2),
        "instrument_positions": static_positions,
        "motion_events":       motion_events,
        "timeline":            timeline,
    }

    print("[Spatial] Timeline built: " + str(len(timeline)) + " time steps")
    return result


def _build_presence_timeline(detections, total_sec, resolution_sec):
    """
    For each time step index, return list of active instruments.
    Uses detection window data — instruments present if detected
    in overlapping window with confidence above threshold.
    """
    n_steps  = int(total_sec / resolution_sec) + 1
    presence = {}

    for dw in detections:
        start_idx = int(dw["start_sec"] / resolution_sec)
        end_idx   = int(dw["end_sec"]   / resolution_sec)

        for det in dw["detections"]:
            if det["confidence"] < 0.15:
                continue
            label = det["label"]
            pos   = INSTRUMENT_POSITIONS.get(label, None)
            if pos is None:
                continue   # skip non-spatial labels

            for idx in range(start_idx, min(end_idx + 1, n_steps)):
                if idx not in presence:
                    presence[idx] = []
                # Add or update
                existing = next((x for x in presence[idx]
                                 if x["label"] == label), None)
                if existing is None:
                    presence[idx].append({
                        "label":      label,
                        "confidence": det["confidence"],
                    })
                elif det["confidence"] > existing["confidence"]:
                    existing["confidence"] = det["confidence"]

    return presence


def _build_motion_lookup(motion_events, total_sec, resolution_sec):
    """
    For each time step index, if a motion event is active return:
    {"dominant_instrument", "spatial_motion", "progress"}
    where progress = 0.0 at event start, 1.0 at event end.
    """
    lookup = {}
    for ev in motion_events:
        start_idx = int(ev["start_sec"] / resolution_sec)
        end_idx   = int(ev["end_sec"]   / resolution_sec)
        duration  = max(ev["end_sec"] - ev["start_sec"], 0.001)

        for idx in range(start_idx, end_idx + 1):
            t_in_event = (idx * resolution_sec) - ev["start_sec"]
            progress   = max(0.0, min(t_in_event / duration, 1.0))
            lookup[idx] = {
                "dominant_instrument": ev["dominant_instrument"],
                "spatial_motion":      ev["spatial_motion"],
                "progress":            progress,
            }

    return lookup


def get_position(instrument_label):
    """Helper to get position for a single instrument label."""
    pos = INSTRUMENT_POSITIONS.get(instrument_label, DEFAULT_POSITION)
    return {"azimuth": pos["azimuth"], "distance": pos["distance"],
            "label": pos.get("label", instrument_label)}


def azimuth_to_description(azimuth):
    """Convert azimuth angle to human-readable direction string."""
    az = azimuth % 360
    if az < 22.5 or az >= 337.5:
        return "front-center"
    elif az < 67.5:
        return "front-right"
    elif az < 112.5:
        return "right"
    elif az < 157.5:
        return "rear-right"
    elif az < 202.5:
        return "rear-center"
    elif az < 247.5:
        return "rear-left"
    elif az < 292.5:
        return "left"
    else:
        return "front-left"
