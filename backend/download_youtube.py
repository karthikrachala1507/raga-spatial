# Raga Spatial - download_youtube.py
# Downloads South Indian instrument clips from YouTube
# Fixed: removed strict duration filter, increased results per query

import os
import json
import subprocess
import sys

import static_ffmpeg
static_ffmpeg.add_paths()

DATA_DIR        = r"D:\raga-spatial-data\raw_stems"
CLIPS_PER_QUERY = 5    # try 5 results per query
TARGET_PER_CAT  = 150  # stop when category reaches this

YOUTUBE_SEARCHES = {
    "percussion": [
        "tabla solo",
        "tabla Indian music",
        "mridangam solo",
        "mridangam concert",
        "kanjira solo",
        "ghatam solo",
        "thavil solo",
        "dholak solo",
        "Indian drum music",
        "south indian percussion music",
        "carnatic percussion",
        "tabla beats instrumental",
        "Indian classical rhythm",
        "pakhawaj solo",
        "dhol drum solo",
    ],
    "plucked_strings": [
        "veena solo",
        "saraswati veena music",
        "sitar solo",
        "sitar music",
        "sarod solo",
        "mandolin Indian music",
        "santoor music",
        "acoustic guitar instrumental",
        "fingerpicking guitar",
        "Indian plucked instrument",
        "veena music classical",
        "sitar raga music",
        "gotuvadyam solo",
        "tanpura music",
        "Indian string instrument music",
    ],
    "bowed_strings": [
        "carnatic violin",
        "Indian violin music",
        "violin raga",
        "violin instrumental Indian",
        "cello solo music",
        "string ensemble music",
        "L Subramaniam violin",
        "violin classical Indian",
        "viola music",
        "orchestral strings music",
    ],
    "wind": [
        "bansuri flute",
        "Indian flute music",
        "nadaswaram music",
        "shehnai music",
        "venu flute",
        "bamboo flute Indian",
        "carnatic flute music",
        "flute instrumental Indian",
        "clarinet instrumental",
        "saxophone instrumental",
        "trumpet solo music",
        "Indian wind instrument",
        "pungi snake charmer flute",
        "South Indian flute",
        "flute raga music",
    ],
    "vocals": [
        "carnatic vocal music",
        "Indian classical singing",
        "South Indian singing",
        "Telugu classical music vocal",
        "Tamil classical singing",
        "carnatic music female vocal",
        "carnatic music male vocal",
        "MS Subbulakshmi music",
        "Telugu folk song",
        "Indian devotional singing",
        "bhajan singing",
        "carnatic alapana vocal",
        "Indian vocal music classical",
        "folk song south india",
        "Telugu song singing",
    ],
    "folk_texture": [
        "dappu drum",
        "Telugu folk music",
        "Telangana folk song",
        "parai drum music",
        "South Indian folk music",
        "tribal music India",
        "Bonalu festival music",
        "dhol drum India",
        "folk percussion India",
        "village music India",
        "Andhra folk music",
        "Tamil folk music",
        "Indian tribal instrument",
        "folk dance music India",
        "urumi melam music",
    ],
    "keys_synth": [
        "harmonium Indian music",
        "harmonium instrumental",
        "piano instrumental music",
        "electric piano music",
        "synthesizer ambient music",
        "tanpura drone music",
        "shruti box music",
        "keyboard instrumental music",
        "organ music instrumental",
        "synth pad music ambient",
        "harmonium bhajan",
        "Indian keyboard music",
        "piano melody music",
        "ambient synthesizer",
        "electronic music ambient",
    ],
    "bass": [
        "bass guitar instrumental",
        "bass guitar solo music",
        "electric bass music",
        "bass guitar groove music",
        "deep bass music",
        "bass guitar funk",
        "bass guitar jazz",
        "808 bass music",
        "sub bass music",
        "bass guitar riff music",
    ],
}


def search_and_download(query, save_dir, max_clips=5):
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir)
                    if f.endswith(".mp3") or f.endswith(".wav")])

    print("[YT] Searching: '" + query + "'")

    search_url = "ytsearch" + str(max_clips) + ":" + query

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extract-audio",
        "--audio-format",   "mp3",
        "--audio-quality",  "128K",
        "--max-filesize",   "100m",
        "--output",         os.path.join(save_dir, "%(title)s_%(id)s.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--ignore-errors",
        search_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        new_count = len([f for f in os.listdir(save_dir)
                         if f.endswith(".mp3") or f.endswith(".wav")])
        downloaded = new_count - existing
        if downloaded > 0:
            print("[YT] +"+str(downloaded)+" clips -- " + query[:45])
        else:
            print("[YT] no new -- " + query[:45])
        return downloaded
    except subprocess.TimeoutExpired:
        print("[YT] timeout -- " + query[:40])
        return 0
    except Exception as e:
        print("[YT] error: " + str(e)[:60])
        return 0


def main():
    print("=" * 60)
    print("Raga Spatial -- YouTube Stem Downloader v2")
    print("=" * 60)

    summary  = {}
    total_dl = 0

    for category, queries in YOUTUBE_SEARCHES.items():
        save_dir       = os.path.join(DATA_DIR, category)
        os.makedirs(save_dir, exist_ok=True)

        # Check existing count
        existing = len([f for f in os.listdir(save_dir)
                        if f.endswith(".mp3") or f.endswith(".wav")])

        print("\n--- " + category + " (have " + str(existing) + ") ---")

        if existing >= TARGET_PER_CAT:
            print("  Already at target. Skipping.")
            summary[category] = existing
            continue

        category_total = 0
        for query in queries:
            current = len([f for f in os.listdir(save_dir)
                           if f.endswith(".mp3") or f.endswith(".wav")])
            if current >= TARGET_PER_CAT:
                print("  Target reached for " + category)
                break
            count          = search_and_download(query, save_dir, CLIPS_PER_QUERY)
            category_total += count
            total_dl       += count

        final_count = len([f for f in os.listdir(save_dir)
                           if f.endswith(".mp3") or f.endswith(".wav")])
        summary[category] = final_count
        print("[" + category + "] Total now: " + str(final_count))

        with open(os.path.join(DATA_DIR, "youtube_progress.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("YOUTUBE DOWNLOAD COMPLETE")
    print("=" * 60)
    grand_total = 0
    for category, count in summary.items():
        print("  " + category.ljust(20) + ": " + str(count) + " clips")
        grand_total += count
    print("  " + "-"*30)
    print("  TOTAL".ljust(22) + ": " + str(grand_total) + " clips")


if __name__ == "__main__":
    main()
