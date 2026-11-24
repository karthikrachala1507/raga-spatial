# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - auto_promote.py
# Decide from SCORECARD.md whether a new model replaces the one the app uses.
# Rule (all must hold): mean recall at least 1 point above the current app model,
# mean flags per song not higher, and no song loses more than 25 recall points.
# On a pass the new checkpoint goes first in MODEL_CANDIDATES in core/separator_8stem.py.
# Usage (from backend\):  python training\auto_promote.py <new tag, e.g. v8> [current tag, default v5]
import re, sys

NEW = sys.argv[1]
CUR = sys.argv[2] if len(sys.argv) > 2 else "v5"
card = open(_os.path.join(paths.DATA_DIR, "evaluation", "reference_songs", "SCORECARD.md"), encoding="utf-8").read()

def overall(tag):
    m = re.search(r"^\| %s \| (\d+)%% \| ([\d.]+) \| \d+ \|" % tag, card, re.M)
    return (float(m.group(1)), float(m.group(2))) if m else None

def per_song(tag):
    out = {}
    for block in card.split("\n## ")[1:]:
        title = block.splitlines()[0]
        m = re.search(r"^\| %s \| (\d+)%% \|" % tag, block, re.M)
        if m:
            out[title] = float(m.group(1))
    out.pop("Overall", None)
    return out

n, c = overall(NEW), overall(CUR)
if not n or not c:
    print("scorecard has no row for", NEW, "or", CUR); sys.exit(2)
pn, pc = per_song(NEW), per_song(CUR)
worst = max(((pc[k] - pn[k], k) for k in pn if k in pc), default=(0, ""))
ok = n[0] >= c[0] + 1 and n[1] <= c[1] and worst[0] <= 25
print("%s recall %.0f%% flags %.1f | %s recall %.0f%% flags %.1f | worst song loss %.0f (%s) -> %s"
      % (NEW, n[0], n[1], CUR, c[0], c[1], worst[0], worst[1][:30], "PROMOTE" if ok else "keep " + CUR))
if ok:
    ck = _os.path.join(paths.CHECKPOINTS_DIR, "separator_%s_best.pt" % NEW)
    f = _os.path.join(paths.BACKEND_DIR, "core", "separator_8stem.py")
    s = open(f, "rb").read().decode("utf-8")
    nl = "\r\n" if "\r\n" in s else "\n"
    line = '    os.path.join(paths.CHECKPOINTS_DIR, "separator_%s_best.pt"),  # promoted automatically by auto_promote.py (%s recall %.0f%% vs %.0f%%)' % (NEW, NEW, n[0], c[0])
    if line.split("#")[0].strip() not in s:
        s = s.replace("MODEL_CANDIDATES = [" + nl, "MODEL_CANDIDATES = [" + nl + line + nl, 1)
        open(f, "wb").write(s.encode("utf-8"))
    print("MODEL_CANDIDATES updated")
