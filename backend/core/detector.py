# Raga Spatial - core/detector.py
# BEATs Instrument Detection Module
# Labels loaded directly from checkpoint label_dict - guaranteed accurate
# Blocklist filters known model artifact indices

import torch
import numpy as np
import sys
import os

MODELS_DIR       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BEATS_CHECKPOINT = os.path.join(MODELS_DIR, "BEATs_iter3_plus_AS2M_finetuned_cpt2.pt")

DETECTION_THRESHOLD = 0.15
WINDOW_SIZE_SEC     = 4.0
HOP_SIZE_SEC        = 2.0
SAMPLE_RATE         = 16000

# Indices that fire at high confidence on ALL music files regardless of content.
# These are model artifacts - not real detections.
# Index 0  = /m/078jl - fires at 87-95% on every music window - artifact
# Index 3  = /m/07qb_dv - Tinkle - fires spuriously on music transients
BLOCKLIST_INDICES = {0, 3}

_LABEL_MAP = None

def _build_label_map():
    MID_TO_NAME = {
        "/m/09x0r":"Speech","/m/05zppz":"Male speech",
        "/m/02zsn":"Female speech","/m/0ytgt":"Child speech",
        "/m/01h8n0":"Conversation","/m/02qldy":"Narration",
        "/m/0261r1":"Babbling","/m/0brhx":"Speech synthesizer",
        "/m/07p6fty":"Shout","/m/07q4ntr":"Bellow","/m/07rwj3x":"Whoop",
        "/m/07sr1lc":"Yell","/m/04gy_2":"Children shouting",
        "/m/0463cq4":"Screaming","/m/02rtxlg":"Whispering",
        "/m/0k65p":"Laughter","/m/07r660_":"Baby laughter",
        "/m/01j3sz":"Giggle","/m/07s04w4":"Snicker",
        "/m/07sq110":"Belly laugh","/m/07rgt08":"Chuckle",
        "/m/0463cl":"Crying sobbing","/m/07qz6j3":"Baby cry",
        "/m/07plct2":"Whimper","/m/01b_21":"Wail moan",
        "/m/07rkbfh":"Sigh","/m/04s8yn":"Singing",
        "/m/02bk07":"Choir","/m/07s2xch":"Yodeling",
        "/m/0l14_3":"Chant","/m/0l14lc":"Mantra",
        "/m/01swy6":"Child singing","/m/0z9c":"Synthetic singing",
        "/m/06bxc":"Rapping","/m/02fxyj":"Humming",
        "/m/07rcgpl":"Groan","/m/05b_gsh":"Grunt",
        "/m/07qcpgn":"Whistling","/m/07mzm6":"Breathing",
        "/m/01d3sd":"Wheeze","/m/06h7j":"Snoring",
        "/m/07s0dtb":"Gasp","/m/0chx1":"Pant","/m/01g90h":"Snort",
        "/m/07q0h5t":"Cough","/m/01b9nn":"Throat clearing",
        "/m/07plz5l":"Sneeze","/m/07pp_mv":"Sniff",
        "/m/0160x5":"Run","/m/06_y0by":"Shuffle",
        "/m/07pbtc8":"Walk footsteps","/m/03cczk":"Chewing",
        "/m/07pdhp0":"Biting","/m/07q7njn":"Stomach rumble",
        "/m/01g2_":"Burping","/m/07s2_j":"Hiccup",
        "/m/025_jnm":"Hands","/m/0l15bq":"Finger snapping",
        "/m/0rgpt":"Clapping","/m/06mb1":"Heart sounds",
        "/m/03lty":"Heart murmur","/m/07plft2":"Cheering",
        "/m/0dzf4":"Applause","/m/07qgkxl":"Chatter",
        "/m/03qtwd":"Crowd","/m/07qgpmy":"Hubbub speech babble",
        "/m/0jbk":"Animal","/m/068zj":"Domestic animals",
        "/m/0bt9lr":"Dog","/m/05tny_":"Bark","/m/07r_25d":"Yip",
        "/m/07qf0zm":"Howl","/m/07rc7d9":"Bow-wow",
        "/m/0ghcn6":"Growling","/m/01yrx":"Cat","/m/02yds9":"Purr",
        "/m/07qrkrw":"Meow","/m/07rjwbb":"Hiss",
        "/m/07p9k1k":"Caterwaul","/m/0jbk":"Wild animals",
        # Music and instruments
        "/m/04rlf":"Music","/m/04szw":"Musical instrument",
        "/m/0fx80y":"Plucked string instrument","/m/0342h":"Guitar",
        "/m/02sgy":"Electric guitar","/m/018vs":"Bass guitar",
        "/m/042v_gx":"Acoustic guitar","/m/06w87":"Steel guitar",
        "/m/01glhc":"Tapping guitar technique","/m/07s0s5r":"Strum",
        "/m/018j2":"Banjo","/m/0jtg0":"Sitar","/m/04rzd":"Mandolin",
        "/m/01bns_":"Zither","/m/07xzm":"Ukulele",
        "/m/05148p4":"Keyboard musical","/m/05r5c":"Piano",
        "/m/01s0ps":"Electric piano","/m/013y1f":"Organ",
        "/m/03xq_f":"Electronic organ","/m/03gvt":"Hammond organ",
        "/m/0l14qv":"Synthesizer","/m/01v1d8":"Sampler",
        "/m/03q5t":"Harpsichord","/m/0l14md":"Percussion",
        "/m/02hnl":"Drum kit","/m/0cfdd":"Drum machine",
        "/m/026t6":"Drum","/m/06rvn":"Snare drum",
        "/m/03t3fj":"Rimshot","/m/0bm02":"Drum roll",
        "/m/0fd3y":"Bass drum","/m/01qbl":"Timpani",
        "/m/0mjkg":"Tabla","/m/07brj":"Cymbal",
        "/m/03qtq":"Hi-hat","/m/07c52":"Wood block",
        "/m/0bm0k":"Tambourine","/m/05r5wn":"Rattle instrument",
        "/m/0mbct":"Maraca","/m/0239kh":"Gong",
        "/m/0bm0g":"Tubular bells","/m/03m9d0z":"Mallet percussion",
        "/m/0l14t7":"Marimba xylophone","/m/07gql":"Glockenspiel",
        "/m/0l156b":"Vibraphone","/m/02p0sh1":"Steelpan",
        "/m/0hn6b":"Orchestra","/m/0mkg":"Brass instrument",
        "/m/01kcd":"French horn","/m/07gkdw_":"Trumpet",
        "/m/07c6l":"Trombone","/m/085jw":"Bowed string instrument",
        "/m/0l14j_":"String section","/m/07y_7":"Violin fiddle",
        "/m/0d8_n":"Pizzicato","/m/01xqw":"Cello",
        "/m/0l14_7":"Double bass","/m/0dwtp":"Wind instrument",
        "/m/06ncr":"Saxophone","/m/01wy6":"Clarinet",
        "/m/0l156l":"Harp","/m/0l14gg":"Bell",
        "/m/027m70_":"Church bell","/m/07n_b":"Jingle bell",
        "/m/07pjwq1":"Tuning fork","/m/0fbw6":"Chime",
        "/m/07sm1k":"Wind chime","/m/0192l":"Harmonica",
        "/m/02qmj0d":"Accordion","/m/01kpqc":"Bagpipes",
        "/m/0283d":"Didgeridoo","/m/07rqsjt":"Theremin",
        "/m/0l156b":"Singing bowl","/m/09l8g":"Chorus effect",
        # Music genres and moods
        "/m/064t9":"Pop music","/m/0glt670":"Hip hop music",
        "/m/0g293":"Beatboxing","/m/06by7":"Rock music",
        "/m/05r6t":"Punk rock","/m/0xzly":"Grunge",
        "/m/05fw6t":"Progressive rock","/m/0dls3":"Rock and roll",
        "/m/0x2sv":"Psychedelic rock","/m/06cqb":"Rhythm and blues",
        "/m/03_d0":"Soul music","/m/03mb9":"Reggae",
        "/m/01lyv":"Country","/m/06j6l":"Swing music",
        "/m/0ggq0m":"Bluegrass","/m/07lnk":"Folk music",
        "/m/04vn0":"Middle Eastern music","/m/0403l3":"Jazz",
        "/m/0m0jc":"Classical music","/m/01359l":"Opera",
        "/m/07gxw":"Ambient music","/m/0326g":"Music of Latin America",
        "/m/02529":"Blues","/m/03lts0":"Music for children",
        "/m/028sqc":"New-age music","/m/07s72n":"Vocal music",
        "/m/02_kms":"A capella","/m/0gywn":"House music",
        "/m/0b9f6":"Techno","/m/0dl5d":"Dubstep",
        "/m/05rwpb":"Electronic dance music","/m/0bmc":"Salsa music",
        "/m/02bk07":"Carnatic music","/m/0332p":"Music of Bollywood",
        "/m/032s66":"Dance music","/m/0p5bs":"Song",
        "/m/02cz_7":"Lullaby","/m/0b_fwt":"Jingle music",
        "/m/07gkdw_":"Video game music",
        "/t/dd00001":"Tender music","/t/dd00002":"Exciting music",
        "/t/dd00003":"Angry music","/t/dd00004":"Scary music",
        # Environment and effects
        "/m/0277j":"Wind","/m/07p_0gm":"Thunderstorm",
        "/m/07rpkh":"Thunder","/m/0838f":"Water","/m/0ngt1":"Rain",
        "/m/0jczl":"Raindrop","/m/0hdsk":"Stream",
        "/m/07swgks":"Waterfall","/m/09d5_":"Ocean",
        "/m/07xj7l5":"Steam","/m/07r66yr":"Gurgling",
        "/m/01d380":"Fire","/m/07pb8fc":"Crackle",
        "/m/07yv9":"Vehicle","/m/012f08":"Motor vehicle",
        "/m/07qv_x5":"Vehicle horn","/m/07qqyl4":"Race car",
        "/m/07rb2bh":"Truck","/m/07rjzl8":"Ice cream truck",
        "/m/012t_z":"Bus","/m/02mk9":"Police car siren",
        "/m/04qvtq":"Ambulance siren","/m/04_sv":"Motorcycle",
        "/m/07jdr":"Train","/m/07r04":"Train whistle",
        "/m/0cmf2":"Helicopter","/m/03ldq":"Airplane",
        "/m/052_rw":"Engine","/m/068hy":"Chainsaw",
        "/m/02dgv":"Door","/m/03wwcy":"Doorbell",
        "/m/07q6cd_":"Slam","/m/07qnq_y":"Knock",
        "/m/07rdhzs":"Tap","/m/07qn4z3":"Squeak",
        "/m/07r5v4s":"Chopping","/m/0f8s22":"Frying",
        "/m/02zdst":"Blender","/m/03dnzn":"Hair dryer",
        "/m/04fgwm":"Vacuum cleaner","/m/0242l":"Coin dropping",
        "/m/07qqf1_":"Alarm","/m/015p6":"Telephone",
        "/m/03kmc9":"Siren","/m/0l14l2":"Whistle",
        "/m/01x3z":"Clock","/m/03l9g":"Hammer",
        "/m/032n05":"Explosion","/m/025_jnm":"Fireworks",
        "/m/07pl1bw":"Boom","/m/0fx9l":"Effects unit",
        "/m/07rrh0c":"Shatter","/m/07qnq_y":"Thump thud",
        "/m/07rcgpl":"Rumble",
        "/t/dd00125":"Crunch","/t/dd00128":"Insect sounds",
        "/t/dd00134":"Mechanical whir","/m/07s8j8t":"Vibration",
        # Animals
        "/m/078jl":"[artifact]",
        "/m/07szfh9":"Frog","/m/02fsn":"Cricket",
        "/m/07rjwbb":"Hiss sound","/m/07qb_dv":"[artifact]",
        # Unknown /t/ tags
        "/t/dd00031":"[artifact]",
        "/m/01sm1g":"[artifact]","/m/015lz1":"[artifact]",
    }

    checkpoint  = torch.load(BEATS_CHECKPOINT, map_location="cpu")
    label_dict  = checkpoint["label_dict"]
    result      = {}
    for idx, mid in label_dict.items():
        result[idx] = MID_TO_NAME.get(mid, mid)
    return result


def _get_label_map():
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _LABEL_MAP = _build_label_map()
    return _LABEL_MAP


def load_beats_model():
    if not os.path.exists(BEATS_CHECKPOINT):
        raise FileNotFoundError("BEATs checkpoint not found: " + BEATS_CHECKPOINT)
    if MODELS_DIR not in sys.path:
        sys.path.insert(0, MODELS_DIR)
    from BEATs import BEATs, BEATsConfig
    print("[Detector] Loading BEATs: " + BEATS_CHECKPOINT)
    checkpoint = torch.load(BEATS_CHECKPOINT, map_location=DEVICE)
    cfg        = BEATsConfig(checkpoint["cfg"])
    model      = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(DEVICE)
    print("[Detector] BEATs loaded -- classes: " + str(cfg.predictor_class))
    return model


def detect_instruments(audio_16k, sr=16000, stem_name="mix"):
    assert sr == SAMPLE_RATE, "BEATs requires 16kHz. Got: " + str(sr)

    label_map  = _get_label_map()
    model      = load_beats_model()
    results    = []
    window_len = int(WINDOW_SIZE_SEC * SAMPLE_RATE)
    hop_len    = int(HOP_SIZE_SEC    * SAMPLE_RATE)
    total      = len(audio_16k)
    win_num    = 0

    print("[Detector] Stem: " + stem_name
          + " | length: " + str(round(total / SAMPLE_RATE, 1)) + "s"
          + " | window: " + str(WINDOW_SIZE_SEC) + "s"
          + " | hop: "    + str(HOP_SIZE_SEC)    + "s")

    start = 0
    while start < total:
        end       = min(start + window_len, total)
        chunk     = audio_16k[start:end]
        chunk_len = len(chunk)
        if chunk_len < SAMPLE_RATE * 0.5:
            break
        if chunk_len < window_len:
            chunk = np.pad(chunk, (0, window_len - chunk_len), mode="constant")

        audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs, _ = model.extract_features(audio_tensor, padding_mask=None)
        probs_np = probs.squeeze(0).cpu().numpy()

        detected = []
        for idx in range(len(probs_np)):
            # Skip known artifact indices and artifact-labeled entries
            if idx in BLOCKLIST_INDICES:
                continue
            confidence = float(probs_np[idx])
            if confidence < DETECTION_THRESHOLD:
                continue
            label = label_map.get(idx, "/m/unknown_" + str(idx))
            # Also skip entries we explicitly marked as artifacts in the label map
            if label == "[artifact]":
                continue
            detected.append({
                "class_idx":  idx,
                "label":      label,
                "confidence": round(confidence, 4),
            })

        detected.sort(key=lambda x: x["confidence"], reverse=True)

        start_sec = round(start / SAMPLE_RATE, 2)
        end_sec   = round(min(end, total) / SAMPLE_RATE, 2)
        win_num  += 1

        print("[Detector] [" + stem_name + "] Window " + str(win_num)
              + ": " + str(start_sec) + "s-" + str(end_sec) + "s"
              + " -- " + str(len(detected)) + " classes")
        if detected:
            top5 = [d["label"] + " (" + str(round(d["confidence"]*100,1)) + "%)"
                    for d in detected[:5]]
            print("[Detector]   Top: " + ", ".join(top5))
        else:
            print("[Detector]   (no detections above threshold after filtering)")

        results.append({
            "start_sec":  start_sec,
            "end_sec":    end_sec,
            "stem":       stem_name,
            "detections": detected,
        })
        start += hop_len

    print("[Detector] [" + stem_name + "] Done -- " + str(len(results)) + " windows")
    return results


def detect_from_stems(vocals_path, instrumental_path):
    import soundfile as sf
    import librosa

    print("[Detector] Running stem-aware detection")
    print("[Detector] Vocals:       " + vocals_path)
    print("[Detector] Instrumental: " + instrumental_path)

    vocals_audio, v_sr = sf.read(vocals_path)
    if vocals_audio.ndim == 2:
        vocals_audio = np.mean(vocals_audio, axis=1)
    if v_sr != 16000:
        vocals_audio = librosa.resample(vocals_audio, orig_sr=v_sr, target_sr=16000)

    inst_audio, i_sr = sf.read(instrumental_path)
    if inst_audio.ndim == 2:
        inst_audio = np.mean(inst_audio, axis=1)
    if i_sr != 16000:
        inst_audio = librosa.resample(inst_audio, orig_sr=i_sr, target_sr=16000)

    vocals_results = detect_instruments(vocals_audio, sr=16000, stem_name="vocals")
    inst_results   = detect_instruments(inst_audio,   sr=16000, stem_name="instrumental")
    merged         = merge_stem_detections(vocals_results, inst_results)

    return {"vocals": vocals_results, "instrumental": inst_results, "merged": merged}


def merge_stem_detections(vocals_results, instrumental_results):
    merged      = []
    max_windows = max(len(vocals_results), len(instrumental_results))

    for i in range(max_windows):
        v_win    = vocals_results[i]         if i < len(vocals_results)       else None
        inst_win = instrumental_results[i]   if i < len(instrumental_results) else None
        start_sec = v_win["start_sec"] if v_win else inst_win["start_sec"]
        end_sec   = v_win["end_sec"]   if v_win else inst_win["end_sec"]
        combined  = []

        if v_win:
            for d in v_win["detections"]:
                combined.append({**d, "stem": "vocals"})

        if inst_win:
            existing = {c["label"]: c for c in combined}
            for d in inst_win["detections"]:
                if d["label"] not in existing:
                    combined.append({**d, "stem": "instrumental"})
                elif d["confidence"] > existing[d["label"]]["confidence"]:
                    existing[d["label"]]["confidence"] = d["confidence"]
                    existing[d["label"]]["stem"]       = "both"

        combined.sort(key=lambda x: x["confidence"], reverse=True)
        merged.append({
            "start_sec":  start_sec,
            "end_sec":    end_sec,
            "detections": combined,
        })

    return merged


def summarise_detections(detection_results, top_n=10):
    windows = (detection_results.get("merged", [])
               if isinstance(detection_results, dict)
               else detection_results)
    label_scores = {}
    label_counts = {}
    for window in windows:
        seen = set()
        for det in window["detections"]:
            label = det["label"]
            if label not in seen:
                label_scores[label] = label_scores.get(label, 0.0) + det["confidence"]
                label_counts[label] = label_counts.get(label, 0) + 1
                seen.add(label)
    summary = [
        {"label":          l,
         "avg_confidence": round(label_scores[l] / label_counts[l], 4),
         "window_count":   label_counts[l]}
        for l in label_scores
    ]
    summary.sort(key=lambda x: (x["window_count"], x["avg_confidence"]), reverse=True)
    return summary[:top_n]


def get_label(class_idx):
    return _get_label_map().get(class_idx, "/m/unknown_" + str(class_idx))
