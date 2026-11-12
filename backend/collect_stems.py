# Raga Spatial - collect_stems.py
# Downloads stem clips from FreeSounds API for all 8 categories
# Uses API key authentication (simpler than OAuth2)

import requests
import os
import time
import json

# FreeSounds API credentials are read from backend/.env (git-ignored) or from
# environment variables. Never hardcode them here. See .env.example.
def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
CLIENT_ID     = os.environ.get("FREESOUND_CLIENT_ID", "")
# The Client Secret is passed as the token= query parameter
CLIENT_SECRET = os.environ.get("FREESOUND_CLIENT_SECRET", "")
API_BASE      = "https://freesound.org/apiv2"

DATA_DIR     = r"D:\raga-spatial-data\raw_stems"
TIER1_TARGET = 200

CATEGORY_QUERIES = {
    "percussion": [
        "tabla solo", "tabla beats", "tabla rhythm",
        "mridangam solo", "mridangam carnatic",
        "kanjira percussion", "ghatam solo",
        "thavil percussion", "dholak beats",
        "indian drum", "south indian percussion",
        "carnatic percussion", "tabla loop",
        "dhol beats", "indian tabla"
    ],
    "plucked_strings": [
        "veena indian", "saraswati veena",
        "sitar solo", "sitar raga",
        "acoustic guitar fingerpicking",
        "acoustic guitar melody",
        "electric guitar clean",
        "sarod solo", "mandolin indian",
        "santoor solo", "indian plucked",
        "guitar riff clean", "veena melody"
    ],
    "bowed_strings": [
        "violin indian classical",
        "carnatic violin",
        "violin solo melodic",
        "cello solo",
        "string ensemble",
        "violin raga",
        "orchestral strings",
        "viola solo",
        "double bass solo",
        "string quartet"
    ],
    "wind": [
        "bansuri flute indian",
        "bansuri raga",
        "venu flute carnatic",
        "nadaswaram",
        "shehnai",
        "flute solo melodic",
        "clarinet solo",
        "saxophone melody",
        "trumpet solo",
        "trombone solo",
        "french horn",
        "indian flute melody",
        "bamboo flute"
    ],
    "keys_synth": [
        "piano solo melodic",
        "piano melody",
        "electric piano rhodes",
        "harmonium indian",
        "organ music",
        "synthesizer pad",
        "synth ambient pad",
        "piano keys",
        "keyboard melody",
        "synth lead",
        "ambient synth",
        "harmonium drone"
    ],
    "vocals": [
        "carnatic vocal",
        "carnatic singing",
        "indian classical vocal",
        "south indian singing",
        "tamil vocal",
        "telugu singing",
        "female indian vocal",
        "male carnatic vocal",
        "folk singing india",
        "indian folk vocal",
        "vocal humming",
        "devotional singing india"
    ],
    "bass": [
        "bass guitar solo",
        "bass guitar groove",
        "electric bass",
        "bass line music",
        "sub bass",
        "deep bass",
        "bass guitar loop",
        "funk bass",
        "bass guitar riff",
        "808 bass",
        "synth bass deep"
    ],
    "folk_texture": [
        "dappu drum telugu",
        "parai drum tamil",
        "dhol folk indian",
        "folk percussion india",
        "indian folk instrument",
        "tribal percussion india",
        "folk drum india",
        "south indian folk",
        "rural india percussion",
        "festival drum india",
        "chimta percussion",
        "kombu instrument"
    ]
}

MIN_DURATION = 3.0
MAX_DURATION = 30.0
MIN_FILESIZE = 50000


def search_sounds(query, page=1, page_size=15):
    """Search FreeSounds using token authentication."""
    url    = API_BASE + "/search/text/"
    params = {
        "query":      query,
        "fields":     "id,name,duration,filesize,previews,license",
        "filter":     "duration:[" + str(MIN_DURATION) + " TO " + str(MAX_DURATION) + "]",
        "page_size":  page_size,
        "page":       page,
        "sort":       "rating_desc",
        "token":      CLIENT_SECRET,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 200:
        return resp.json()
    else:
        print("[Search] Error " + str(resp.status_code) + " for: " + query)
        return None


def download_sound(preview_url, save_path):
    """Download a sound preview file."""
    params = {"token": CLIENT_SECRET}
    resp   = requests.get(preview_url, params=params,
                          stream=True, timeout=30)
    if resp.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False


def collect_category(category, queries, target_count, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing   = len([f for f in os.listdir(save_dir)
                      if f.endswith(".mp3") or f.endswith(".wav")])
    downloaded = existing

    print("\n[" + category + "] Have: " + str(existing)
          + " / target: " + str(target_count))

    if existing >= target_count:
        print("[" + category + "] Already at target. Skipping.")
        return existing

    for query in queries:
        if downloaded >= target_count:
            break

        print("[" + category + "] Query: '" + query + "'")
        page = 1

        while downloaded < target_count:
            results = search_sounds(query, page=page)
            if not results or not results.get("results"):
                break

            for sound in results["results"]:
                if downloaded >= target_count:
                    break

                duration = sound.get("duration", 0)
                filesize = sound.get("filesize", 0)

                if duration < MIN_DURATION or duration > MAX_DURATION:
                    continue
                if filesize < MIN_FILESIZE:
                    continue

                previews = sound.get("previews", {})
                url      = previews.get("preview-hq-mp3") or \
                           previews.get("preview-lq-mp3")
                if not url:
                    continue

                filename  = category + "_" + str(sound["id"]) + ".mp3"
                save_path = os.path.join(save_dir, filename)

                if os.path.exists(save_path):
                    continue

                if download_sound(url, save_path):
                    downloaded += 1
                    name = sound.get("name", "")[:50]
                    print("[" + category + "] "
                          + str(downloaded) + "/" + str(target_count)
                          + " — " + name)

                time.sleep(0.3)

            if results.get("next"):
                page += 1
            else:
                break

    print("[" + category + "] Done: " + str(downloaded) + " clips")
    return downloaded


def main():
    print("=" * 60)
    print("Raga Spatial — Stem Collection (FreeSounds API)")
    print("Target: " + str(TIER1_TARGET) + " clips per category")
    print("Save dir: " + DATA_DIR)
    print("=" * 60)

    if not CLIENT_SECRET:
        print("ERROR: FREESOUND_CLIENT_SECRET is not set.")
        print("Copy backend\\.env.example to backend\\.env and fill in your key.")
        return

    # Test API connection first
    print("\n[Auth] Testing API connection...")
    test = search_sounds("piano", page_size=1)
    if test is None:
        print("ERROR: API connection failed.")
        print("Check your CLIENT_SECRET key.")
        return
    print("[Auth] API connection OK")

    summary = {}
    for category, queries in CATEGORY_QUERIES.items():
        save_dir = os.path.join(DATA_DIR, category)
        count    = collect_category(category, queries,
                                    TIER1_TARGET, save_dir)
        summary[category] = count
        with open(os.path.join(DATA_DIR, "progress.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    total = 0
    for cat, count in summary.items():
        print("  " + cat.ljust(20) + ": " + str(count))
        total += count
    print("  TOTAL".ljust(22) + ": " + str(total))


if __name__ == "__main__":
    main()
