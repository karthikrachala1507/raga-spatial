# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - motion_vs_real.py
# Does the app's motion detector fire where the ORIGINAL song really pans an instrument?
# For each song: run the pipeline with motion on, take the detected events (per stem, with times),
# measure that stem's real pan every second in the stereo original (masks applied to L and R),
# and compare the real pan change inside the events with the change outside them.
# Output: data\evaluation\reference_songs\MOTION_VS_REAL.md
# Usage (from backend\):  python training\motion_vs_real.py [song-name-part ...]

import os, sys, subprocess, tempfile
import numpy as np
import soxr

from core import separator_8stem as sep
from core.renderer import run_full_pipeline
from training.analyze_reference_songs import SONGS, OUT_DIR, find_song, ffmpeg_exe
from training.stereo_motion_check import load_stereo, SR
from training.stem_stereo_motion import pan_series


def main():
    only = [a.lower() for a in sys.argv[1:]] or ["insecurities", "peydhavi", "kanave", "3 bgm", "unakku"]
    dev = sep.pick_device()
    model, _ = sep.load_model(dev)
    lines = ["# Detector events vs real panning", "",
             "Per detected motion event: the real pan change of that stem in the original stereo mix during the event "
             "(max-min of the 1 s pan values inside it) against the same measure for equally long stretches outside events.", "",
             "| song | stem | events | real pan change inside events | outside events | verdict |", "|---|---|---|---|---|---|"]
    for part, _ in SONGS:
        if not any(o in part.lower() for o in only):
            continue
        path = find_song(part)
        if not path:
            continue
        name = os.path.basename(path)[:30]
        wav = os.path.join(tempfile.gettempdir(), "mvr_" + str(abs(hash(part))) + ".wav")
        subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", path, "-ac", "2", "-ar", "44100", wav], check=True)
        res = run_full_pipeline(wav, "mvr_" + str(abs(hash(part)) % 10**6), motion=True)
        evs = res.get("motion_events", [])
        print(name, len(evs), "events", flush=True)
        if not evs:
            lines.append("| %s | - | 0 | - | - | no events |" % name)
            continue
        st = load_stereo(path)
        mid = st.mean(axis=0)
        est = sep.infer_16k(model, soxr.resample(mid, SR, sep.MODEL_SR).astype(np.float32), dev)
        sl, _ = sep.fullband_stems(st[0], est, SR)
        sr_, _ = sep.fullband_stems(st[1], est, SR)
        by = {}
        for e in evs:
            by.setdefault(e.get("stem") or e.get("instrument") or e.get("carrier") or "?", []).append(e)
        for stem, es in by.items():
            if stem not in model.categories:
                lines.append("| %s | %s | %d | (stem name not found; keys: %s) | | |" % (name, stem, len(es), list(es[0].keys())))
                continue
            ci = model.categories.index(stem)
            pan, en = pan_series(sl[ci], sr_[ci])
            loud = en > 0.05 * np.median(en + 1e-12)
            inside = np.zeros(len(pan), bool)
            ch_in = []
            for e in es:
                a, b = int(e["start_sec"]), int(np.ceil(e["end_sec"]))
                seg = slice(a, min(b, len(pan)))
                if seg.stop - seg.start >= 2 and loud[seg].all():
                    ch_in.append(float(pan[seg].max() - pan[seg].min()))
                inside[seg] = True
            L = max(2, int(np.median([e["end_sec"] - e["start_sec"] for e in es])))
            ch_out = [float(pan[i:i + L].max() - pan[i:i + L].min()) for i in range(0, len(pan) - L, L)
                      if loud[i:i + L].all() and not inside[i:i + L].any()]
            a_in = float(np.mean(ch_in)) if ch_in else float("nan")
            a_out = float(np.mean(ch_out)) if ch_out else float("nan")
            verdict = "follows real panning" if a_in > 1.3 * a_out else "no link to real panning"
            lines.append("| %s | %s | %d | %.2f | %.2f | %s |" % (name, stem, len(es), a_in, a_out, verdict))
    text = "\n".join(lines)
    open(os.path.join(OUT_DIR, "MOTION_VS_REAL.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
