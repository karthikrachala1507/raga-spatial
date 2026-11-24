# --- project paths (see backend/paths.py) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import paths

# Raga Spatial - corpus_gate.py  (second gate after auto_promote.py)
# Reverts an automatic promotion if the larger corpus benchmark says the new model is worse
# than the one it replaced: recall more than 1 point lower, or more than 0.05 more flags per song.
# Usage (from backend\):  python training\corpus_gate.py <new tag> [old tag, default v5]
import re, sys

NEW = sys.argv[1]
OLD = sys.argv[2] if len(sys.argv) > 2 else "v5"
md = open(_os.path.join(paths.DATA_DIR, "evaluation", "corpus", "CORPUS_BENCHMARK.md"), encoding="utf-8").read()

def row(tag):
    m = re.search(r"^\| %s \| ([\d.]+)%% \| ([\d.]+) \|" % tag, md, re.M)
    return (float(m.group(1)), float(m.group(2))) if m else None

n, o = row(NEW), row(OLD)
f = _os.path.join(paths.BACKEND_DIR, "core", "separator_8stem.py")
s = open(f, "rb").read().decode("utf-8")
marker = "separator_%s_best.pt" % NEW
promoted = any(marker in ln and "auto_promote" in ln for ln in s.splitlines())
if not n or not o:
    print("corpus benchmark has no row for", NEW, OLD); sys.exit(2)
bad = n[0] < o[0] - 1 or n[1] > o[1] + 0.05
print("corpus: %s recall %.1f flags %.2f | %s recall %.1f flags %.2f | promoted=%s -> %s"
      % (NEW, n[0], n[1], OLD, o[0], o[1], promoted, "REVERT" if (bad and promoted) else "ok"))
if bad and promoted:
    out = [ln for ln in s.splitlines(True) if not (marker in ln and "auto_promote" in ln)]
    open(f, "wb").write("".join(out).encode("utf-8"))
    print("promotion reverted")
