# Raga Spatial - tests/test_api.py
# End-to-end check of the web API: starts the server, uploads the test song,
# waits for the pipeline, then checks every route the UI uses.
#
#   python tests\test_api.py
#
# Takes about a minute on the GPU. Stop training first if it is running (the
# separator needs the GPU memory).

import os
import sys
import time
import subprocess
import requests

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8791
BASE = "http://127.0.0.1:%d/api" % PORT
results = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""), flush=True)
    results.append(ok)
    return ok


def main():
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT)],
        cwd=_BACKEND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    job = None
    try:
        for _ in range(60):
            try:
                if requests.get(BASE + "/health", timeout=3).ok:
                    break
            except Exception:
                time.sleep(1)
        health = requests.get(BASE + "/health", timeout=30).json()
        check("server starts, health ok", health.get("status") == "ok", str(health.get("status")))
        mode = requests.get(BASE + "/mode", timeout=10).json()
        check("mode endpoint reports the 8-stem model", mode.get("mode") == "8-stem", str(mode.get("mode")))

        with open(os.path.join(_BACKEND, "tests", "test_song.flac"), "rb") as f:
            up = requests.post(BASE + "/upload", files={"file": ("test_song.flac", f)}, timeout=60).json()
        job = up["job_id"]
        check("upload accepted", bool(job), job)

        t0 = time.time()
        status = {}
        while time.time() - t0 < 600:
            status = requests.get(BASE + "/status/" + job, timeout=10).json()
            if status["status"] in ("complete", "error"):
                break
            time.sleep(3)
        check("pipeline completes", status.get("status") == "complete",
              "%s after %.0f s" % (status.get("status"), time.time() - t0))
        if status.get("status") != "complete":
            print("message:", status.get("message"))
            return

        res = requests.get(BASE + "/result/" + job, timeout=30).json()
        check("result is 8-stem, pure or hybrid", res.get("mode") == "8-stem" and res.get("separation_mode") in ("pure", "hybrid"),
              "%s / %s" % (res.get("mode"), res.get("separation_mode")))
        stems = res.get("stems8", [])
        check("result lists 9 stems with layers 1-9", [s["layer"] for s in stems] == list(range(1, 10)))
        active = [s["name"] for s in stems if s["active"]]
        check("at least 3 stems are active", len(active) >= 3, str(active))
        named = [i["label"] for s in stems for i in s["instruments"]]
        check("instrument names are readable (no raw /m/ codes)", not any(n.startswith("/m/") for n in named),
              ", ".join(sorted(set(named))[:6]))

        r = requests.get(BASE + "/download/" + job, timeout=60)
        check("binaural mix downloads", r.ok and r.headers.get("content-type", "").startswith("audio") and len(r.content) > 1e6,
              "%.1f MB" % (len(r.content) / 1e6))
        r = requests.get(BASE + "/stem/%s/%s" % (job, active[0]), timeout=60)
        check("per-stem binaural file downloads", r.ok and len(r.content) > 1e5, active[0])
        r = requests.get(BASE + "/download/%s/stems/%s" % (job, active[0]), timeout=60)
        check("dry stem downloads", r.ok and len(r.content) > 1e6)
        r = requests.get(BASE + "/stem/%s/folk_texture_x" % job, timeout=10)
        check("unknown stem name is rejected", r.status_code == 400, str(r.status_code))
        r = requests.get(BASE + "/download/..%5C..%5Cetc", timeout=10)
        check("path-like job id is rejected", r.status_code in (400, 404), str(r.status_code))
    finally:
        if job:
            try:
                requests.delete(BASE + "/jobs/" + job, timeout=30)
            except Exception:
                pass
        server.terminate()
    print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
