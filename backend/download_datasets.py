# Raga Spatial - download_datasets.py
# Downloads open stem datasets from Zenodo into D:\raga-spatial-data\datasets
# and extracts only the audio the separator needs (isolated stems).
#
# Usage:
#   python download_datasets.py                 (everything, in priority order)
#   python download_datasets.py saraga_carnatic (one or more by name)
#
# Safe to stop and rerun: downloads resume, finished datasets are skipped.
# Licences: several of these are NON-COMMERCIAL / research-only. See DATASETS.md.

import os
import sys
import time
import hashlib
import tarfile
import zipfile
import requests

DATASETS_DIR = r"D:\raga-spatial-data\datasets"
API = "https://zenodo.org/api/records/"
CHUNK = 4 * 1024 * 1024


def keep_musdb(name):
    return name.lower().endswith(("vocals.wav", "bass.wav", "drums.wav"))


def keep_saraga(name):
    n = name.lower()
    return ".multitrack-" in n and n.endswith(".mp3")


def keep_urmp(name):
    base = os.path.basename(name)
    return base.startswith("AuSep_") and base.lower().endswith(".wav")


def keep_wav(name):
    return name.lower().endswith(".wav")


def keep_slakh(name):
    n = name.replace("\\", "/")
    return n.endswith("metadata.yaml") or ("/stems/" in n and n.endswith(".flac"))


# name, Zenodo record, files, which members to keep
DATASETS = [
    {"name": "saraga_carnatic", "record": 4301737,
     "files": ["saraga1.5_carnatic.zip"], "keep": keep_saraga},
    {"name": "guitarset", "record": 3371780,
     "files": ["audio_mono-mic.zip"], "keep": keep_wav},
    {"name": "mridangam_tani", "record": 1266207,
     "files": ["mridangam_tani-avarthanam_1.0.zip"], "keep": keep_wav},
    {"name": "musdb18hq", "record": 3338373,
     "files": ["musdb18hq.zip"], "keep": keep_musdb},
    {"name": "urmp", "record": 5034983,
     "files": ["Dataset.tar.gz"], "keep": keep_urmp},
    {"name": "medley_solos", "record": 3464194,
     "files": ["Medley-solos-DB_metadata.csv", "Medley-solos-DB.tar.gz"], "keep": keep_wav},
    # Slakh2100 is 104 GB. Files inside the archive are NOT grouped by track,
    # so it has to be downloaded whole (resumable). Only stems + metadata are
    # extracted (no mixes or MIDI) and the archive is deleted afterwards.
    {"name": "slakh2100", "record": 4599666,
     "files": ["slakh2100_flac_redux.tar.gz"], "keep": keep_slakh},
]


def log(msg):
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)


def record_files(record):
    r = requests.get(API + str(record), timeout=60)
    r.raise_for_status()
    out = {}
    for f in r.json().get("files", []):
        out[f["key"]] = {
            "url": f["links"]["self"],
            "size": f["size"],
            "md5": f.get("checksum", "").replace("md5:", ""),
        }
    return out


def download(url, dest, size):
    """Resumable download with retries."""
    for attempt in range(1, 31):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= size:
            return True
        try:
            headers = {"Range": "bytes=" + str(have) + "-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code != 206:
                    have = 0                      # server ignored Range: start over
                    open(dest, "wb").close()
                r.raise_for_status()
                last = time.time()
                with open(dest, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 120:
                            log("    %.1f / %.1f GB" % (have / 1e9, size / 1e9))
                            last = time.time()
        except Exception as e:
            log("    connection problem (%s), retry %d" % (type(e).__name__, attempt))
            time.sleep(min(60, 5 * attempt))
    return os.path.getsize(dest) >= size


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def safe_target(root, member_name):
    target = os.path.normpath(os.path.join(root, member_name))
    if not target.startswith(os.path.normpath(root) + os.sep):
        return None                                  # refuse paths escaping root
    return target


def extract_zip(path, root, keep):
    n = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir() or not keep(info.filename):
                continue
            target = safe_target(root, info.filename)
            if target is None:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target + ".part", "wb") as dst:
                for block in iter(lambda: src.read(CHUNK), b""):
                    dst.write(block)
            os.replace(target + ".part", target)
            n += 1
    return n


def extract_tar(fileobj_or_path, root, keep, max_tracks=None):
    """Extract kept members. Works on a file or on a live HTTP stream."""
    n = 0
    tracks = set()
    if isinstance(fileobj_or_path, str):
        tar = tarfile.open(fileobj_or_path, mode="r|*")
    else:
        tar = tarfile.open(fileobj=fileobj_or_path, mode="r|*")
    last = time.time()
    with tar:
        for member in tar:
            if not member.isfile():
                continue
            if max_tracks is not None:
                parts = member.name.replace("\\", "/").split("/")
                track = next((p for p in parts if p.startswith("Track")), None)
                if track:
                    tracks.add(track)
                    if len(tracks) > max_tracks:
                        log("    reached %d tracks, stopping the stream" % max_tracks)
                        break
            if not keep(member.name):
                continue
            target = safe_target(root, member.name)
            if target is None:
                continue
            if os.path.exists(target) and os.path.getsize(target) == member.size:
                continue                              # already have it (rerun)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tar.extractfile(member)
            with open(target + ".part", "wb") as dst:
                for block in iter(lambda: src.read(CHUNK), b""):
                    dst.write(block)
            os.replace(target + ".part", target)
            n += 1
            if time.time() - last > 120:
                log("    extracted %d files (%d tracks)" % (n, len(tracks)))
                last = time.time()
    return n


def fetch(ds):
    root = os.path.join(DATASETS_DIR, ds["name"])
    done = os.path.join(root, ".done")
    os.makedirs(root, exist_ok=True)
    if os.path.exists(done):
        log(ds["name"] + ": already done, skipping")
        return True

    files = record_files(ds["record"])
    for key in ds["files"]:
        if key not in files:
            log(ds["name"] + ": file not found on Zenodo: " + key)
            return False
        info = files[key]

        if ds.get("stream"):
            log("%s: streaming %s (%.1f GB total, keeping first %d tracks)" % (
                ds["name"], key, info["size"] / 1e9, ds["max_tracks"]))
            for attempt in range(1, 6):
                try:
                    with requests.get(info["url"], stream=True, timeout=120) as r:
                        r.raise_for_status()
                        n = extract_tar(r.raw, root, ds["keep"], ds["max_tracks"])
                    log("%s: stream finished, %d new files" % (ds["name"], n))
                    break
                except Exception as e:
                    log("    stream problem (%s: %s), attempt %d" % (type(e).__name__, e, attempt))
                    time.sleep(30)
            else:
                log(ds["name"] + ": stream kept failing; keeping what was extracted")
            continue

        archive = os.path.join(root, key)
        log("%s: downloading %s (%.2f GB)" % (ds["name"], key, info["size"] / 1e9))
        if not download(info["url"], archive, info["size"]):
            log(ds["name"] + ": download incomplete, rerun to resume")
            return False
        if info["md5"]:
            if md5_of(archive) != info["md5"]:
                log(ds["name"] + ": CHECKSUM MISMATCH for " + key + " - deleting, rerun to retry")
                os.remove(archive)
                return False
            log(ds["name"] + ": checksum OK")

        low = key.lower()
        if low.endswith(".zip"):
            n = extract_zip(archive, root, ds["keep"])
        elif low.endswith((".tar.gz", ".tgz", ".tar")):
            n = extract_tar(archive, root, ds["keep"])
        else:
            n = 0                                     # plain file (e.g. csv): keep as is
        if low.endswith((".zip", ".tar.gz", ".tgz", ".tar")):
            log("%s: extracted %d audio files, removing the archive" % (ds["name"], n))
            os.remove(archive)

    with open(done, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(ds["name"] + ": DONE")
    return True


def main():
    wanted = sys.argv[1:]
    todo = [d for d in DATASETS if not wanted or d["name"] in wanted]
    unknown = set(wanted) - {d["name"] for d in DATASETS}
    if unknown:
        print("Unknown dataset(s): " + ", ".join(sorted(unknown)))
        print("Available: " + ", ".join(d["name"] for d in DATASETS))
        return
    os.makedirs(DATASETS_DIR, exist_ok=True)
    for ds in todo:
        try:
            fetch(ds)
        except Exception as e:
            log("%s: FAILED (%s: %s)" % (ds["name"], type(e).__name__, e))
    log("ALL REQUESTED DATASETS PROCESSED")


if __name__ == "__main__":
    main()
