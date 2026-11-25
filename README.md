# Raga Spatial

Spatial (binaural) audio for South Indian film music. Upload a song; the app
splits it into 8 instrument stems, places each stem at its own direction and
depth around the listener, and plays the result on headphones with a live
map and a stem mixer.

This file is the full record of the project: what every part does, what
was tried, what failed and why, what data is used, and where things stand.
It is written so that a person or an AI assistant can pick the project up
without any other context.

**Last updated: 13 November 2026.**

---

## Contents

1. [Status at a glance](#1-status-at-a-glance)
2. [Quick start](#2-quick-start)
3. [Folder layout](#3-folder-layout)
4. [How the app works](#4-how-the-app-works)
5. [The 8 stems and where they sit](#5-the-8-stems-and-where-they-sit)
6. [Models used](#6-models-used)
7. [Training the 8-stem separator](#7-training-the-8-stem-separator)
8. [Training data](#8-training-data)
9. [Separator versions: v1, v2, v3](#9-separator-versions-v1-v2-v3)
10. [Experiments and measurements](#10-experiments-and-measurements)
11. [Project history](#11-project-history)
12. [Errors met and how they were fixed](#12-errors-met-and-how-they-were-fixed)
13. [Open problems](#13-open-problems)
14. [The Folk / Texture problem](#14-the-folk--texture-problem)
15. [What is left to do](#15-what-is-left-to-do)
16. [Environment](#16-environment)
17. [Notes for AI assistants](#17-notes-for-ai-assistants)
18. [Glossary](#18-glossary)

---

## 1. Status at a glance

| Part | State |
|---|---|
| 2-stem app (vocals + instrumental) | Works. Kept as a fallback. |
| 8-stem backend (separation, positions, depth, rendering) | Works end to end. Tested on a 30 s song and a 5 min song. |
| 8-zone UI with live stem mixer | Works. Checked in a browser. |
| Training pipeline | Works. Pause and resume are safe. |
| Training data | 9 open datasets (incl. Saraga Audiovisual) plus the older FreeSounds/YouTube clips; all converted. Folk / texture clips cleaned. Coherent (same-song) stems cut for all 5 multitrack datasets. |
| **8-stem model (v3)** | **Trained: 100 of 100 epochs (finished 18 Nov).** Best checkpoint epoch 93: random mixes 3.50 dB, coherent mixes 6.62 dB. Five stems are at 6 dB or better on coherent mixes; keys is weak (1.5 dB). Not studio quality. |
| Training | **Finished.** Nothing is running. |
| Slakh conversion | Done (20,676 of 20,676 stems). |
| Coherent cutting | Done for URMP, MUSDB18-HQ, Saraga Carnatic, Slakh (900 tracks) and Saraga Audiovisual. |

**The honest summary:** all the software is built and tested. The model that
does the actual separating has barely started learning, so stems from a real
song still have heavy bleed. It needs days of training. One data problem
([section 14](#14-the-folk--texture-problem)) was fixed on 13 November; the
model has not been trained on the cleaned data yet, so the fix is unproven.

**How to read the dB numbers** (SDR, explained in the [glossary](#18-glossary)):

| SDR | Meaning |
|---|---|
| 0 dB | no better than silence |
| 1-3 dB | starting to separate, heavy bleed between stems |
| 4-6 dB | usable for spatial placement |
| 7+ dB | good |

---

## 2. Quick start

The two things done most often need no terminal:

| What | How |
|---|---|
| Start the app | double-click `start_app.bat`, then use `http://localhost:3000` |
| Start or resume training | double-click `start_training.bat` |
| Finish preparing data, then train | double-click `start_data_then_training.bat` (use this while datasets are still being converted) |
| Pause training | Ctrl+C in its window. It resumes from the last finished epoch. |

For everything else, open a terminal and go to the backend folder first:

```cmd
conda activate raga-spatial
cd /d D:\raga-spatial\backend
```

| What | Command |
|---|---|
| Score the current model in dB | `python training\evaluate_validation.py` |
| Write the 8 stems of a real song, to listen to | `python training\evaluate_separator.py --song "D:\path\song.mp3"` |
| Download the open datasets | `python data_tools\download_datasets.py` |
| Turn downloaded datasets into training clips | `python data_tools\prepare_datasets.py` |
| Cut multitrack datasets into time-aligned stems | `python data_tools\prepare_coherent.py` |
| Quick audio checks (about 1 minute) | `python tests\test_8stem.py` |
| Whole pipeline on the test song | `python tests\test_8stem.py pipeline` |
| Web API end to end (server, upload, every route) | `python tests\test_api.py` |

`download_datasets.py`, `prepare_datasets.py` and `prepare_coherent.py` can be
stopped and rerun at any time; they skip what is already done. Training picks up new clips at the
start of each epoch.

A finished test result is kept on disk. With the app running, open
`http://localhost:3000/?job=t8test` to see and hear it.

---

## 3. Folder layout

Everything lives in one folder, `D:\raga-spatial`.

```
raga-spatial\
├── start_app.bat              double-click: starts the app and opens the browser
├── start_training.bat         double-click: start or resume training
├── start_data_then_training.bat   finish converting datasets, then train
├── README.md                  this file
├── DATASETS.md                dataset sources and licences
│
├── backend\
│   ├── paths.py               the one file that says where everything lives
│   ├── main.py  api\          the web server (FastAPI) and its routes
│   ├── requirements.txt       package versions the project runs on
│   ├── .env                   FreeSounds key (not in git; see .env.example)
│   ├── core\                  the audio pipeline
│   │   ├── preprocess.py        load audio, tempo, onsets, chroma
│   │   ├── separator_2stem.py   MelBandRoformer: vocals + instrumental
│   │   ├── separator_8stem.py   the custom model: 8 full-bandwidth stems
│   │   ├── bsrnn.py             the 8-stem model itself
│   │   ├── detector.py          BEATs: names the instruments in each stem
│   │   ├── audioset_labels.csv  Google's public AudioSet class names (fills the label gaps)
│   │   ├── motion.py            finds melodic runs that should move in space
│   │   ├── spatial.py           positions by instrument label (2-stem path)
│   │   ├── spatial_8stem.py     the 8 positions, depth layers and rendering
│   │   ├── hrtf.py              binaural (HRTF) rendering and distance effects
│   │   └── renderer.py          runs the whole pipeline for one song
│   ├── training\
│   │   ├── train_separator.py     trains the 8-stem model
│   │   ├── evaluate_validation.py scores a checkpoint in dB
│   │   └── evaluate_separator.py  writes the 8 stems of a real song
│   ├── data_tools\
│   │   ├── collect_stems.py       Tier 1: FreeSounds clips
│   │   ├── download_youtube.py    Tier 1: YouTube audio
│   │   ├── verify_stems.py        Tier 1: cut raw files into clips
│   │   ├── download_datasets.py   Tier 2: open datasets from Zenodo
│   │   ├── prepare_datasets.py    Tier 2: cut datasets into clips
│   │   ├── prepare_coherent.py    cut multitrack datasets into time-aligned stems
│   │   ├── clean_folk_texture.py  keeps only folk clips that are percussion
│   │   └── augment_mix.py         legacy: writes fixed mixes (no longer needed)
│   ├── tests\                 test_8stem.py, test_api.py, the phase tests, test_song.flac
│   ├── models\                MelBandRoformer, BEATs and HRTF files used by the app
│   └── outputs\               what the app produces for each uploaded song
│
├── frontend\                  the React UI
│   └── src\  App.jsx  Stems8.jsx  App.css  main.jsx
│
├── docs\                      private notes, not in git (the June project export)
│
└── data\                      everything large, not in git
    ├── raw_stems\             Tier 1 raw downloads (FreeSounds, YouTube)
    ├── datasets\              Tier 2 open datasets, stems only
    ├── processed\stems_t1\    Tier 1 training clips
    ├── processed\stems_t2\    Tier 2 training clips
    ├── processed\stems_aligned\  same-song, same-moment stems for coherent mixes
    ├── processed\rejected\    clips removed from training (kept, not deleted)
    ├── checkpoints\           trained separator (separator_v3_best.pt is used by the app)
    ├── logs\                  training_log_v1/v2/v3.json, one entry per epoch
    ├── real_songs\            full songs for listening tests (test5.mp3)
    └── evaluation\            stems written by evaluate_separator.py
```

Two model folders exist on purpose: `backend\models\` holds downloaded
models the app needs; `data\checkpoints\` holds what training produces.

No script contains a hardcoded data path. They all import `backend\paths.py`,
so the project can be moved by moving the folder. To keep the data
elsewhere, set the environment variable `RAGA_DATA_DIR`.

---

## 4. How the app works

The frontend uploads a song to the backend. `core\renderer.py` runs seven
steps and reports progress. Which path it takes depends on whether a trained
8-stem model exists.

### 8-stem path (used when `data\checkpoints\separator_v3_best.pt` exists)

```
song
 │ 1. preprocess      load as mono 44.1 kHz, normalise, tempo, beats, onsets
 │ 2. separate        8 stems (see below)
 │ 3. detect          BEATs on each stem: names the instruments inside it
 │ 4. motion          on plucked strings and wind: find melodic runs
 │ 5. position        each stem gets its direction and depth layer;
 │                    moving stems get a path around the listener
 │ 6. render          distance effect, then HRTF per stem, then sum
 │ 7. export          JSON for the UI
 ▼
binaural mix (WAV) + one binaural file per stem (MP3) + 8 dry stems (WAV) + JSON
```

**Step 2 in detail.** There are two modes:

- **pure (default):** all 8 stems come from the custom model.
- **hybrid:** MelBandRoformer extracts the vocals and the custom model splits
  the rest into 7 stems. Set `RAGA_SEP8_MODE=hybrid`. **Not recommended**: on
  18 Nov 2026 the MelBandRoformer checkpoint was found to load with 842
  missing and 120 unexpected weights (it does not match its architecture), and
  in hybrid mode the 8 stems no longer add back up to the song (error 0.15
  instead of 0.0004). The custom model's own vocals reach 6.5 dB on coherent
  mixes.

**Full-bandwidth trick.** The custom model works at 16 kHz mono, which on its
own would cut every stem off at 8 kHz. Instead, its output is used only to
decide what share of each time-frequency cell belongs to each stem. Those
shares (soft masks that add up to 1) are applied to the original 44.1 kHz
audio. Above 7.5 kHz, where the model cannot see, each stem gets the share it
holds in the 4 to 7.5 kHz band at that moment. Two consequences: stems keep
the full bandwidth, and the 8 stems always add back up to the original song.

**Step 6 in detail.** For each stem: lower the level and roll off the highs
according to its distance, convolve with the HRTF for its direction, add a
diffuse reverb tail that grows with distance. Moving stems are rendered in
overlapping blocks, each with the HRTF of its own direction.

### 2-stem path (fallback when no 8-stem model exists)

MelBandRoformer gives vocals and instrumental. BEATs labels the mix. Each
label has a fixed position. This was the whole app before November 2026.
Its limit: every instrument sits in one "instrumental" stem at one position.

### Other switches

| Environment variable | Effect |
|---|---|
| `RAGA_SEP8_MODE=hybrid` | use MelBandRoformer for the vocals (off by default, see section 4) |
| `RAGA_FORCE_2STEM=1` | use the 2-stem path even if an 8-stem model exists |
| `RAGA_SKIP_DETECT=1` | skip BEATs instrument naming (faster) |
| `RAGA_DEVICE=cpu` or `cuda` | force where the 8-stem model runs |
| `RAGA_SEP8_MODEL=<path>` | use a specific checkpoint |
| `RAGA_DATA_DIR=<path>` | keep the data folder somewhere else |

By default the 8-stem model uses the GPU only if at least 3 GB is free, so
the app still works (on CPU, slower) while training is running.

### API routes (`backend\api\routes.py`, prefix `/api`)

| Route | Purpose |
|---|---|
| `POST /upload` | upload a song, start the pipeline, returns a job id |
| `GET /status/{job}` | progress, step 0 to 7 |
| `GET /result/{job}` | the result JSON (also works after a server restart) |
| `GET /download/{job}` | the binaural mix (WAV) |
| `GET /download/{job}/stems/{name}` | one dry stem (WAV) |
| `GET /stem/{job}/{name}` | one binaural stem (MP3), used by the live mixer |
| `GET /mode` | whether the next upload will use 8-stem or 2-stem |
| `GET /jobs`, `DELETE /jobs/{job}`, `GET /health` | housekeeping |

### Frontend (`frontend\src`)

- `App.jsx`: upload, progress, the 2-stem result view, motion timeline.
  A finished job can be reopened with `?job=<id>` in the address bar.
- `Stems8.jsx`: the 8-stem view. A map with 8 depth rings where each stem
  is a dot that grows with its level and follows its motion path, and a
  mixer with volume, mute, solo and download per stem. All binaural stems are
  decoded into memory and played together through the Web Audio API, so the
  mixer acts instantly and the stems stay sample-aligned.

---

## 5. The 8 stems and where they sit

Direction: 0° is front, 90° right, 180° rear, 270° left.
Depth: 0 is nearest, 1 is farthest. Layer 1 is closest to the listener.

| Layer | Stem (`name`) | Contains | Direction | Depth |
|---|---|---|---|---|
| 1 | Vocals (`vocals`) | voice, humming, chorus | 0° front | 0.20 |
| 2 | Plucked strings (`plucked_strings`) | veena, sitar, guitar, mandolin, santoor | 315° front-left | 0.35 |
| 3 | Wind (`wind`) | bansuri, nadaswaram, shehnai, flute, sax, brass | 45° front-right | 0.45 |
| 4 | Keys / synth (`keys_synth`) | piano, harmonium, synth, tanpura drone | 30° front-right | 0.50 |
| 5 | Bowed strings (`bowed_strings`) | violin, viola, cello, string section | 225° rear-left | 0.60 |
| 6 | Percussion (`percussion`) | tabla, mridangam, ghatam, kanjira, drums, thavil | 180° rear | 0.65 |
| 7 | Bass (`bass`) | bass guitar, 808, sub | 135° rear-right | 0.75 |
| 8 | Folk / texture (`folk_texture`) | dappu, parai, dhol, kombu, atmosphere | 270° left | 0.85 |

Set in `backend\core\spatial_8stem.py` (`STEM_LAYOUT`).

**Distance effect** (`core\hrtf.py`), for depth `d`:

- level: `1 / (1 + 4d)`
- high-frequency roll-off: low-pass at `18000 - 12000d` Hz
- reverb: mix of `0.3d`, with a longer and later tail when farther

Bass, bowed strings, percussion and folk get a small level trim (+1 to +4 dB)
so that pushing them back does not thin the mix. Bass gets no low-pass and no
reverb.

**Motion.** Plucked strings and wind may sweep around the listener during
melodic runs: left to right, right to left, or a full circle. The detector
looks at pitch movement (chroma) and note density of that stem.

Why 8 category stems and not one per instrument: separating 132 individual
instruments was judged infeasible on a 6 GB GPU. Instead, each category is
one stem, and BEATs names the instruments inside it.

---

## 6. Models used

| Model | File | Job | Notes |
|---|---|---|---|
| MelBandRoformer | `backend\models\MelBandRoformer.ckpt` | vocals + instrumental at 44.1 kHz | dim 384. Downloaded, not trained here. |
| BEATs | `backend\models\BEATs_iter3_plus_AS2M_finetuned_cpt2.pt` | names instruments (527 AudioSet classes) | Western bias; Carnatic vocals are sometimes labelled "Speech". |
| MIT KEMAR HRTF | `backend\models\mit_kemar_normal_pinna.sofa` | binaural rendering | 710 directions, 512 taps, 44.1 kHz. |
| Custom band-split RNN (v3) | `data\checkpoints\separator_v3_best.pt` | the 8 stems | Trained in this project. 9.5M parameters. |

`backend\models\model_bs_roformer_ep_317_sdr_12.9755.ckpt` is also on disk.
It is **not used**: it has dim 512 and cannot load into the dim 384 model.
It caused a recurring error (see [section 12](#12-errors-met-and-how-they-were-fixed)).
It can be deleted.

### The custom model (`core\bsrnn.py`)

A standard band-split RNN (Luo and Yu, 2022), sized for a 6 GB GPU:

1. The spectrogram (16 kHz, FFT 1024, hop 256) is cut into 40 bands: 16 bands
   of 4 bins up to 1 kHz, 8 of 8 bins to 2 kHz, 8 of 16 bins to 4 kHz, 8 of 32
   bins to 8 kHz. Each band is normalised and projected to 112 numbers.
2. Eight blocks alternate an LSTM along time (how each band evolves) with an
   LSTM across bands (how bands relate at one moment, which is what tells a
   flute from a violin).
3. Every band keeps its own feature vector to the end and predicts its own
   complex mask for each of the 8 stems.

---

## 7. Training the 8-stem separator

Script: `backend\training\train_separator.py`. Start with `start_training.bat`.

### How training data is built

Mixes are built **on the fly**, and the target stems are always exactly what
went into the mix. There are two kinds, used half and half:

- **Random mixes.** Pick 2 to 6 categories, a recording in each, a clip from
  that recording, a random level for each, and add them up. Every combination
  is new, but the parts are unrelated: different pieces, keys and tempos.
- **Coherent mixes.** Take one moment of one real multitrack recording and use
  the stems that actually played together, at their true balance (each stem's
  level is nudged by up to 3 dB for variety). Half the time one or two
  unrelated clips are added for stems that recording lacks, because several
  datasets have no vocals or only three stems. These mixes are what real
  songs look like: parts that move together are much harder to pull apart.

Coherent mixes need `data_tools\prepare_coherent.py` to have been run. Until
then training uses random mixes only and says so when it starts.

- Clips are drawn **per source recording**, not per clip, so a category with
  a few very long recordings does not drown out the others.
- **No dataset may dominate a stem.** Each dataset's share of a stem follows
  the square root of its number of recordings. Before this, Slakh, which is
  synthesised, supplied 72 to 95% of five stems.
- 10% of source recordings are **held out** and never trained on.
- Validation uses a fixed, seeded set of 300 random mixes built only from
  held-out recordings, so the score measures real generalisation and is the
  same after every resume.
- A second validation set uses **coherent mixes from held-out tracks**
  (`coherent SDR` in the output). It is the closest thing to a real song
  that can still be scored. A held-out track is never trained on in any
  form, not even as single clips.
- New clips placed in the data folders are picked up at the next epoch.

### Settings

| Setting | Value |
|---|---|
| Sample rate, clip length | 16 kHz mono, 10 s clips, random 6 s window per step |
| Batch size | 4 |
| Mixes per epoch | 12,000 (3,000 steps) |
| Epochs | up to 100 |
| Optimiser | Adam, learning rate 0.001, multiplied by 0.98 each epoch |
| Mixed precision | on (about 4.4 GB of GPU memory) |
| Data workers | 4 |
| Loss | L1 on the waveform + 0.5 × L1 on spectrogram magnitude for stems that are present; 0.05 × L1-to-zero for stems that are absent |

### What it prints and saves

Each epoch prints the training loss, the validation loss, `SDR: x dB` and a
line of per-stem dB values. It saves:

- `separator_v3_latest.pt`: everything needed to resume
- `separator_v3_best.pt`: the epoch with the best validation SDR; this is
  what the app loads
- `separator_v3_epochN.pt`: every 10 epochs
- `data\logs\training_log_v3.json`: one entry per epoch

"Best" is chosen by separation quality in dB (the average of the random and
coherent scores when both exist), not by the loss.

### Speed

About 30 minutes per epoch when nothing else is using the D: drive. About 95
minutes per epoch when the Slakh conversion runs at the same time, because
both read heavily from the same disk. Run them one after the other.

---

## 8. Training data

Licences and sources are in `DATASETS.md`. All clips are 10 s, 16 kHz mono
WAV, in one folder per category.

### Tier 1: FreeSounds and YouTube (`data\processed\stems_t1`)

Collected in May 2026 with `collect_stems.py` (FreeSounds previews, 128 kbps)
and `download_youtube.py` ("solo" searches), then cut with `verify_stems.py`
(10 s windows, 5 s overlap, peak-normalised).

| Category | Raw files | Clips | Source recordings |
|---|---|---|---|
| percussion | 158 | 9,106 | 83 |
| plucked_strings | 211 | 569 | 151 |
| bowed_strings | 258 | 245 | 64 |
| wind | 201 | 202 | 81 |
| keys_synth | 203 | 423 | 159 |
| vocals | 107 | 15,698 | 86 |
| bass | 207 | 268 | 104 |
| folk_texture | 93 | 1,612 (was 8,163) | 48 (was 84) |
| **Total** | **1,438** | **28,123** | **776** |

Weaknesses: clip counts are inflated by overlapping windows from few long
recordings; YouTube "solos" often contain accompaniment; FreeSounds previews
are lossy. The `folk_texture` clips were whole songs; 6,551 of them were
moved out of training on 13 November
([section 14](#14-the-folk--texture-problem)). The other categories have not
been checked the same way: a sample of Tier 1 percussion clips had a voice in
8% of them.

### Tier 2: open datasets (`data\processed\stems_t2`)

Downloaded with `download_datasets.py` from Zenodo (no sign-up), cut with
`prepare_datasets.py` (non-overlapping 10 s windows, quiet windows skipped, a
cap per recording).

| Dataset | Download | Clips so far | Feeds |
|---|---|---|---|
| Slakh2100 | 104 GB | 47,492 (65% converted) | plucked, bowed, wind, keys, bass, percussion |
| Saraga Carnatic 1.5 | 14.4 GB | 11,419 | Carnatic vocals, violin, mridangam, ghatam |
| MUSDB18-HQ | 22.7 GB | 4,917 | vocals, bass, drums |
| Medley-solos-DB | 7.9 GB | 2,090 | wind, violin, piano, guitar, voice |
| URMP | 12.1 GB | 1,349 | solo bowed strings and wind |
| GuitarSet | 0.7 GB | 994 | acoustic guitar |
| Mridangam Tani-avarthanam | 0.3 GB | 376 | solo mridangam |
| 4-way Tabla (ISMIR 2021) | 0.2 GB | 544 | tabla recorded in isolation |

Tier 2 totals by category:

| Category | Clips | Source recordings |
|---|---|---|
| percussion | 11,261 | 1,886 |
| plucked_strings | 13,542 | 3,716 |
| bowed_strings | 9,901 | 1,682 |
| wind | 6,592 | 1,614 |
| keys_synth | 14,761 | 3,908 |
| vocals | 5,422 | 333 |
| bass | 7,158 | 1,552 |
| folk_texture | 0 | 0 |

Notes:

- Saraga is recorded live, so its stems contain some bleed from the other
  players. Only the loud windows of each stem are kept.
- Slakh is synthesised from MIDI with sample libraries. Its General MIDI
  program numbers decide the category (in `prepare_datasets.py`).
- **Licences:** Saraga, MUSDB18-HQ and the mridangam set are non-commercial.
  A model trained on them is fine for research or a demo, not for a paid
  product.

### Time-aligned stems for coherent mixes (`data\processed\stems_aligned`)

Cut by `prepare_coherent.py` from the four datasets that hold the separate
parts of the same performance. Every stem of a track is cut at the same 10 s
windows, and all stems of a window share one gain, so their real balance is
kept. Only windows where at least two different stems are playing are used.

| Dataset | State | Result |
|---|---|---|
| URMP | done | 44 pieces read, 78 windows; only 7 pieces mix strings with wind, the rest are one family only |
| MUSDB18-HQ | not cut yet | vocals + bass + drums of the same song |
| Saraga Carnatic | not cut yet | voice + violin + mridangam of the same concert |
| Slakh2100 | not cut yet | up to 900 tracks, about 10 instruments each |

### Audit of 13 November 2026

| Check | Result |
|---|---|
| Broken clips (250 per stem per tier) | none unreadable, wrong rate, wrong length or silent |
| Mix equals the sum of its target stems | yes, to within 0.0000002 |
| Recordings shared by training and validation | 0 |
| Hours per stem | percussion 58, vocals 59, keys 46, plucked 42, bowed 30, bass 22, wind 20, **folk 4.5** |
| Who supplied each stem before balancing | Slakh: keys 95%, plucked 89%, wind 82%, bass 82%, bowed 72%, percussion 60% |
| After balancing | real recordings: percussion 65%, bowed 48%, wind 42%, plucked 38%, bass 35%, keys 22% |
| Wrong content, by the detector | Tier 1 percussion: a voice in 15% of clips, no percussion heard in 27%. The clean datasets: 0 to 3%. |

The detector cannot judge two things: it mistakes bowed strings for voices
(92% of clean solo violin clips), and it misses Carnatic singing (voice heard
in only 17% of Saraga vocal clips). A check of the vocal clips with
MelBandRoformer was started and stopped unfinished after 45 minutes.

### Still missing

- `folk_texture`: no open dataset exists for dappu, parai or dhol. It has
  4.5 hours from 48 recordings, by far the least.
- A search of Zenodo on 13 November found no open stem data for bansuri,
  nadaswaram, shehnai, veena, sitar or harmonium. The only usable find was
  the isolated tabla set, which was added.
- Veena, nadaswaram, thavil, bansuri and harmonium have no clean multitrack
  data. Clean solo recordings of these would help the most.

### Adding your own stems

Put 10 s, 16 kHz mono WAVs in `data\processed\stems_t2\<category>\`, named
`<set>__<source>__<number>.wav`. Clips from the same recording must share
the `<set>__<source>` part. Training picks them up at its next epoch.

---

## 9. Separator versions: v1, v2, v3

| | v1 (May to June) | v2 (12 Nov) | v3 (13 Nov, current) |
|---|---|---|---|
| Network | 64 separate band LSTMs → one global LSTM → 8 mask decoders ending in Tanh | same, plus a learnable mask gain | standard band-split RNN: 40 bands, time and band LSTMs, per-band masks |
| Parameters | 144.2M | 144.2M | 9.5M |
| Loss | `-SDR + 0.1 × L1` | L1 + 0.5 × spectral L1 on present stems, 0.05 × L1-to-zero on absent | same as v2 |
| Training mixes | 5,000 fixed mixes on disk | built on the fly | built on the fly |
| Target stems | re-normalised to peak 0.9 (wrong level) | exactly what is in the mix | exactly what is in the mix |
| Validation | random 10% of mixes (same clips as training) | held-out Tier 1 recordings | held-out recordings from Tier 1 and the finished Tier 2 sets |
| "Best" chosen by | loss | loss | SDR in dB |
| Time per epoch | about 1.8 hours | about 30 min alone (75 with downloads running) | about 30 min alone |
| Trained for | 100 epochs, 16 days | 4 epochs, 5 hours | 3 epochs so far |
| Result | **silence** (output RMS about 0.00003) | **bass only**: bass 5.7 dB, every other stem under 1 dB | median 1.3 dB; bass 5.5, vocals 2.8, folk 2.5, wind 1.9 |
| Fate | checkpoints deleted | checkpoints deleted | training paused after epoch 3 |

### Why v1 failed

Its SDR loss used the **target's** energy as the numerator, where it should
have used the energy of the estimate projected onto the target. Shrinking
the output shrank the error term and so raised the score. The loss rewarded
silence, and it went down for 100 epochs while the output faded to nothing.
A second flaw made it worse: the saved target stems were louder than their
true level in the mix (see [section 10](#10-experiments-and-measurements)).

Warning signs that were missed: the loss was positive for the whole run
(mean SDR negative), and nobody listened to an output until the end.

### Why v2 was replaced

After the loss and data were fixed, v2 trained without collapsing but learned
only bass, which can be separated by frequency alone. Its design pushed all
513 frequency bins × 8 stems of every frame through a single 1024-number
vector, and its 64 band LSTMs never exchanged information except through
that vector. Telling instruments apart needs exactly that cross-band
information. v3 uses the published design that v2 was loosely based on.

### v3 so far (validation, held-out recordings)

| Epoch | Median | Bass | Vocals | Folk | Wind | Plucked | Keys | Bowed | Percussion |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.06 | 3.77 | 2.18 | 1.24 | 1.24 | 1.00 | 0.72 | 0.30 | 0.08 |
| 2 | 1.32 | 5.27 | 2.68 | 2.31 | 1.98 | 1.00 | 0.51 | 0.37 | 0.13 |
| 3 | 1.28 | 5.49 | 2.75 | 2.50 | 1.89 | 0.68 | 0.43 | 0.29 | 0.12 |

Training loss was still falling (0.193, 0.178, 0.167). The median stalled
between epochs 2 and 3. Three epochs are too few to conclude anything;
percussion, bowed strings and keys had not started to separate.

**This first run was set aside on 13 November.** Since it started, the
training data changed in four ways (folk clips cleaned, datasets balanced,
tabla added, coherent mixes added), and the model had learned to put whole
songs into Folk / Texture. Its resume file is kept in
`data\checkpoints\v3_run1\` and its log in `data\logs\training_log_v3_run1.json`.
The next start trains from zero. Until its first epoch finishes, the app
keeps using the old run's `separator_v3_best.pt`.

---

## 9b. The final model (v3, trained 13-18 November 2026)

100 epochs, about 27 minutes each, 44 hours of training. Scores are SDR in dB
on held-out recordings (0 = silence, 4 to 6 = usable, 7+ = good).

| Epoch | Random mixes | Coherent mixes |
|---|---|---|
| 1 | 0.84 | 3.01 |
| 7 | 1.49 | 4.81 |
| 40 | 2.74 | 6.07 |
| 93 (best, used by the app) | **3.50** | 6.62 |
| 100 (last) | 3.35 | **6.64** |

Final per stem (epoch 100):

| Stem | Random mixes | Coherent mixes |
|---|---|---|
| Wind | 5.43 | 12.59 |
| Bass | 3.45 | 7.41 |
| Percussion | 2.10 | 7.19 |
| Vocals | 6.53 | 6.49 |
| Bowed strings | 3.13 | 6.13 |
| Plucked strings | 3.13 | no coherent data |
| Folk / texture | 2.69 | no coherent data |
| Keys | 1.47 | no coherent data |

The scores rose fast until about epoch 40, then crept: coherent +0.5 and
random +0.8 over the last 60 epochs.

**Real-song test (`test5.mp3`, 5 minutes, epoch 93, custom model alone).**
Share of the song's energy per stem: vocals 45.5%, bass 38.7%, percussion
5.2%, bowed strings 5.3%, folk / texture **4.2%** (it was 64% before the
folk clips were cleaned), keys 1.1%, plucked strings 0.0%, wind 0.0%.
The Folk / Texture problem is fixed. Two things are still unverified by ear:
plucked strings and wind at 0% (absent from the song, or missed?), and bass at
39% (it may be absorbing the low part of other instruments). Listen to the 8
files in `data\evaluation\test5`.

**App pipeline test (30 s song, pure mode).** All checks pass: the 8 stems add
back up to the song (error 0.0004), 5 of 8 stems active, 45 seconds on the
GPU. BEATs named Guitar and Synthesizer in the right stems, and "Yodeling"
and "Drum machine" for vocals and percussion.

## 9c. Verification of 22 November 2026

Everything below was run, from the final layout, after training finished.

| Check | Result |
|---|---|
| Audio checks (`test_8stem.py`) | all pass |
| Whole pipeline, pure mode, 30 s song | all pass: stems add back up to the song (error 0.0004), 45 s on the GPU |
| Whole pipeline, hybrid mode | **fails** the sum check (error 0.15); see section 4. Off by default. |
| Web API end to end (`test_api.py`): health, mode, upload, status, result, mix, per-stem and dry-stem downloads, rejection of bad stem names and path-like job ids | 13 of 13 pass; the pipeline took 33 s |
| Phase tests `test_separator.py`, `test_detector.py`, `test_renderer.py` | pass (they had not been run since the reorganisation) |
| Earlier phase tests (`test_spatial`, `test_hrtf`, `test_motion`, `test_preprocess`) | pass (13 Nov) |
| Saraga Audiovisual against Saraga 1.5 for duplicate recordings | none: 8 shared song titles out of 212, all with different lengths (different performances). No validation leakage from this. |
| Instrument names | 291 of the 527 detector labels were showing as raw codes such as `/m/01p970`. All now have names (AudioSet class list added). |

Two small bugs found by these checks and fixed: `core/preprocess.py` printed a
check-mark character that crashes on a Windows console, and the 2-stem separator
crashed on audio shorter than 16 s (13 Nov).

## 9d. Reference-song analysis (23 November 2026)

The owner named 11 songs as references and what each should teach (bass,
vocals, instrument separation and placement, motion, flute / veena / violin,
overall balance). `backend	rainingnalyze_reference_songs.py` measured each
one. Nothing was trained: the owner chooses which characteristics to train.
Numbers are in `data\evaluationeference_songs\`; the songs are not copied.

| Finding | Evidence |
|---|---|
| The model is lopsided on real film songs | Vocals plus bass take 70-90% of most songs. Wind, plucked strings (veena) and keys get 0-4% even where they are clearly present. |
| Heavy electronic sub-bass is misread | "Ordinary Person" has 44% of its energy below 60 Hz; the model gave 77% to Folk / Texture and only 6% to Bass. Training bass is real and synthesised bass guitar, not 808-style sub-bass. |
| Bass is almost mono in electronic songs, wide in live ones | Left-right correlation below 250 Hz is 0.95-1.0 in Achcham Telugandham, Dheema, Ordinary Person, 3 bgm; 0.2-0.4 in Kanave, Peydhavi, Sajra, Insecurities. |
| Motion is found only when plucked or wind stems carry the tune | 0 events in Hello Rammante (including its first 30 s), Infatuation and Achcham Telugandham; 4-15 events in Kanave, Insecurities, Peydhavi, Unakku Thaan, 3 bgm. |
| Instrument names stay generic | "Guitar", "Organ", "Synthetic singing". Violin or cello is named in 5 songs; flute, veena and mridangam are never named (BEATs has no such classes). |
| Balance song | "Sajra savera" (the owner's balance reference) comes out 76% vocals, 13% Folk / Texture (tabla), 1-3% for everything else. |

## 9e. Real-song scorecard and fine-tunes v4 to v6 (23-24 November 2026)

`backend/training/scorecard_reference_songs.py` compares what each model finds
in the 11 reference songs with what is in them. "What is in them" is BEATs run
on the original mix plus `data/evaluation/reference_songs/truth.json` (the
assistant's best guess, unverified; correct it where you know better). Output:
`data/evaluation/reference_songs/SCORECARD.md`. Fine-tunes start from an
earlier checkpoint through environment variables (`RAGA_INIT_FROM`,
`RAGA_LOSS_NORM`, `RAGA_DROP_PROB` and others, see the `start_finetune_*.bat`
files).

| Model | What changed | Mean recall | Flags per song |
|---|---|---|---|
| v3 | 100-epoch model | 61% | 2.3 |
| v4 | plus synthetic 808 bass | 55% | 2.5 (the validation gain did not carry over to real songs) |
| v5 | real multitrack windows, mastering chain (EQ, reverb, bus compression) | 69% | 1.9 (**used by the app**) |
| v6 | level-normalised loss, absent-stem penalty 0.3, vocals/bass dropped from 30% of windows | 72% | 1.7 (not promoted) |

v6 helps Ordinary Person and Kun Faya Kun but loses bowed strings on Kanave-Kanave
and 3 bgm, so v5 stays in the app. Vocals still take 62-84% of Kun Faya Kun,
Peydhavi and Unakku Thaan, and bass is a sink in Dheema, Hello Rammante and
3 bgm. The loss weighting is not the main cause: the training data has too few
clean South Indian vocal, tabla and bass examples.

Trap: set `NUMBA_CACHE_DIR` to an existing writable folder
(`data/numba_cache`) or analysis hangs at preprocessing.

### v7, the 9th stem, placement planner and motion toggle (24 November 2026)

- **9th stem, `hand_percussion`** (tabla, mridangam, ghatam), split from drums by
  file or source name when the training pool is built (no files moved). The model
  head is widened from v5 (`start_finetune_9stem.bat`). Old 8-stem checkpoints
  still load (`categories` is stored in the checkpoint).
- **v7 result:** mean recall 65%, 2.0 flags per song, below v5 (69%) and v6 (72%),
  so it is **not promoted**; v5 stays in the app. It lost bowed strings on
  Peydhavi and went catch-all on Ordinary Person, and did not recover tabla on
  Kun Faya Kun (86% vocals). More data, not more fine-tuning, is the next lever.
- **Placement planner** (`plan_layout` in `spatial_8stem.py`): stems closer than
  35 degrees and 0.20 depth are moved apart, the front 20 degrees stay for vocals,
  and absent stems free their space.
- **Motion is optional**, off by default (checkbox in the UI, `motion` form field
  on upload). `stereo_motion_check.py` measures real panning in the original mixes
  (`STEREO_MOTION.md`): the balance changes over time in many songs, but this may
  be arrangement rather than one instrument moving.
- **Data:** access to the Sanidha dataset (vocals, violin, mridangam, ghatam) was
  requested by email on 24 November 2026; no reply yet.

### Motion that follows the original recording (24 November 2026)

The rule-based detector (rising/falling phrases) did not line up with any real
panning in the originals (`MOTION_VS_REAL.md`), so the optional motion feature now
**measures** movement instead (`backend/core/real_motion.py`):

1. The separator's masks are applied to the left and right channel of the original
   separately, giving each stem's own left/right balance every second.
2. A stem counts as moving only if its pan spread is at least 0.20 and it swings by
   more than 0.25 inside 4 s for at least half of the time it is audible (stem share
   at least 5%). Vocals and bass are never moved.
3. The measured pan, relative to the stem's usual position, becomes an azimuth offset
   (60 degrees per unit of pan, at most 70 degrees) around the planned position.

Calibration on synthetic mixes with known pan paths (`SYNTHETIC_MOTION.md`, 10 s
clips, 8-12 mixes per stem): a still stem is wrongly flagged 8-12% of the time, a swept
stem is caught 50-75% of the time (rule is deliberately conservative). On the real
songs it is selective: Unakku Thaan moves bowed strings only; Peydhavi moves wind and
keys; Kanave moves strings. Motion stays off by default. The old detector is no longer
used by the app. Trap: scripts run with the env's python.exe directly need
`<env>in` on PATH (or an activated env), or BEATs fails with a missing
`nvrtc-builtins64_118.dll`.


## 10. Experiments and measurements


Every number here was measured, not estimated.

| Question | Method | Result |
|---|---|---|
| Were v1's target stems at the right level? | 300 of the old mixes: fit each mix as a weighted sum of its saved stems | Fit was exact (residual 0.01%). Targets were a median **7 dB too loud** (worst 19 dB); 61% were more than 2× too loud. The mask could boost at most about 3 dB, so most targets were unreachable. |
| Does mixed precision help? | same data, on and off | 1.59 vs 1.92 s per step; 4.6 vs 6.8 GB of GPU memory (the card has 6.4 GB). |
| Do data workers help? | 2 workers vs none | 1.60 vs 3.89 s per step while the disk was busy. |
| Is the v2 model separating? (epoch 3 and 4) | SDR on held-out mixes | Tier 1 held-out: 0.1 dB median. Clean Tier 2 held-out: 0.7 dB median, bass 5.7 dB, all others under 1 dB. |
| Would compressing v2's input help? | 3 variants, 250 steps each, same seed and data | No difference: loss curves matched to four decimals. |
| Is the standard band-split RNN better? | 1,200 steps (11 minutes), then scored on the same held-out mixes | Median 1.4 dB, bass 8.5, wind 2.8, vocals 2.2. More in 11 minutes than v2 reached in 5 hours, at a third of the time per step. |
| How big can it be? | dim 96×6, 128×6, 128×8, 112×8 at batch 4 | 128×8 needs 5.0 GB (too tight). Chosen: dim 112 × 8 blocks, 4.4 GB. |
| Does the moving-source renderer add tremolo? | envelope ripple of moving vs static noise | 0.112 vs 0.114: none. |
| Do the 8 stems add back up to the song? | sum of stems minus input | Largest difference 0.0002 on a full-scale signal. |
| What do real songs look like to the epoch 2 model? | `test5.mp3` (5 min) | folk_texture 64%, bass 21%, vocals 7.5%, everything else under 5%. See [section 14](#14-the-folk--texture-problem). |
| What is inside the folk_texture clips? | BEATs on all 8,163 clips | 20% clean percussion; 13% percussion with a voice or melody; 27% voice or melody without percussion; 35% nothing musical recognised; 5% weak percussion. |
| How long does the app take? | epoch 2 model on the GPU | 30 s song: 26 s for the whole pipeline. 5 min song: 29 s to separate and write the 8 stems. |

---

## 11. Project history

### May 2026: the 2-stem app (Phases 1 to 11)

| Date | Event |
|---|---|
| 18 May | Initial commit. |
| 19 May | Phases 1-4: FastAPI backend, preprocessing, MelBandRoformer separation. |
| 22-23 May | Phases 5-10: BEATs detector with label map, motion detection (three rewrites), spatial positions, HRTF rendering, full pipeline, API. |
| 23-24 May | Phase 11: React UI. The frontend had to be rebuilt from a clean Vite scaffold. The 30 s test ran end to end: 120.2 BPM, 3 motion events. |

Complaints after Phase 11 that drove everything since: the output sounded
"meshy" or noisy, and **bass and most instruments never appeared** on the
map. Cause: BEATs only labels the mix; it does not separate. All instruments
sat in one stem at one position.

### Late May 2026: the pivot to 8 stems

- Demucs (4 or 6 Western stems) was rejected as not covering South Indian
  instruments. Decision: train a custom 8-category separator.
- Tier 1 data collected: 1,154 FreeSounds clips, then YouTube audio, 1,438
  raw files in total; 34,674 clips after cutting (30 May).
- 5,000 fixed training mixes generated (31 May).

### June 2026: v1

- 31 May to 16 June: 100 epochs, with a nine-day pause between epochs 30 and
  40. Validation loss fell from 8.38 to 3.78.
- 18 June: evaluation on a real song. Every stem was silent. A v2 training
  script was written the same morning and never run.

The project then sat untouched for five months.

### 12 November 2026

1. Project state rebuilt from the June export; the disk confirmed v2 had
   never been run.
2. The stem-level error in `augment_mix.py` was measured (7 dB) and fixed,
   along with three resume bugs in the training script.
3. All uncommitted work was committed and pushed, including the whole
   frontend, which had existed only on the laptop.
4. A FreeSounds key was found hardcoded in `collect_stems.py` and already on
   GitHub. It was moved to a git-ignored `.env` file.
5. Seven open datasets were chosen and started downloading.
6. Training was reworked to build mixes on the fly. v2 training started.
7. The 8-stem backend and the 8-zone UI were written and tested.

### 13 November 2026

1. v2 measured after 4 epochs: bass only.
2. Input-compression test: no effect. Band-split RNN trial: clearly better.
3. v3 training started at 03:55. Epochs 1 to 3 done by 07:38.
4. Old v1 and v2 checkpoints and the old mix folders deleted (28 GB).
5. Training and the Slakh conversion paused at the owner's request.
6. Everything moved into one folder, `D:\raga-spatial`; the old copy on C:
   was deleted after a file-by-file comparison.
7. Files and folders renamed for clarity; every reference updated and
   verified; this README written.
8. The Folk / Texture problem diagnosed and the clips cleaned: 1,612 kept,
   6,551 moved out of training.
9. Full data audit. Sampler changed so no dataset dominates a stem; isolated
   tabla recordings added; a crash on short audio fixed.
10. Coherent mixes added: a tool that cuts multitrack datasets in matching
    time windows, a second validation set from held-out tracks, and a
    sampler that uses coherent mixes for half of training. Tested on URMP.
11. The first v3 run set aside so the next start trains on the corrected data.

### Commits

| Commit | Date | What |
|---|---|---|
| `59247da` to `37ec04f` | 18-23 May | Phases 1 to 10 |
| `90d0392` | 31 May | Tier 1 data tools and v1 training script |
| `c541077` | 12 Nov | v2 script, stem-level fix, resume fixes |
| `53fb17e` | 12 Nov | 2-stem separator: checkpoint order, soft clip |
| `97ed3dc` | 12 Nov | React frontend |
| `f417071` | 12 Nov | FreeSounds key moved to `.env` |
| `937d53b` | 12 Nov | on-the-fly training, dataset tools, 8-stem backend and UI |
| `cd1379a` | 13 Nov | validation scorer, app launcher, dataset notes |
| `91bf1ab` | 13 Nov | v3: standard band-split RNN |
| `6e7b884` | 13 Nov | one project folder, central paths, tools and tests folders |
| `fe499e8` | 13 Nov | clearer names, model in `core\bsrnn.py` |

---

## 12. Errors met and how they were fixed

### Model and training

| Problem | Cause | Fix |
|---|---|---|
| v1 output was silent after 100 epochs | SDR loss rewarded shrinking the output | Loss replaced (L1 + spectral L1). Output level is now monitored every epoch. |
| Target stems 7 dB too loud | `augment_mix.py` re-normalised each saved stem to peak 0.9 instead of saving what went into the mix | Stems are saved at their level in the mix; training now builds mixes itself with correct levels. |
| Train/validation split changed on every resume | unseeded shuffle | Validation is a fixed, seeded set from held-out recordings. |
| Training log wiped on every resume | log list reset at start | Earlier epochs are reloaded. (v1's log only has epochs 72-100 for this reason.) |
| "Best" value saved one epoch stale | saved before it was updated | Updated first, then saved. |
| Validation overstated quality | same clips in training and validation | 10% of source recordings held out. |
| A few long recordings dominated a category | sampling by clip | Sampling by source recording. |
| GPU memory over the limit (6.8 of 6.4 GB) | full-precision 144M model | Mixed precision; v3 is also far smaller. |
| Epochs slow (3.9 s per step) | reading clips from a busy disk in the main process | Data worker processes. |
| v2 learned only bass | bottleneck in the network design | v3: standard band-split RNN. |
| One synthesised dataset supplied 72-95% of five stems | sampling followed the number of recordings, and Slakh has the most | Each dataset's share now follows the square root of its size. |
| Training mixes were correct but not musical | every mix combined unrelated clips | Half the mixes are now coherent: one moment of one real multitrack recording. |
| Half-written checkpoint or clip could be read | files written in place | Written to a temporary name, then renamed. |

### App and audio

| Problem | Cause | Fix |
|---|---|---|
| "Meshy", noisy output | the moving-source renderer faded every 0.1 s chunk in and out, a 10 Hz tremolo | Overlap-add rendering with windows that sum to 1. |
| Bass and most instruments never shown | detection on the mix, no separation | The 8-stem path. Bass now appears as its own stem, labelled "Bass guitar". |
| Size mismatch loading the separator (recurred several times) | the dim 512 BSRoformer checkpoint was tried for a dim 384 model | Only `MelBandRoformer.ckpt` is loaded; the fallback is removed. |
| Cracking at chunk boundaries (2-stem) | chunk edges | Hann overlap-add and a soft clip. Partly effective. |
| BEATs labels were garbage | wrong label mapping | Labels read from the checkpoint; two artefact classes blocked. |
| Motion detection found 0 events, then one 30 s blob | absolute thresholds | Relative threshold (mean + 0.03). |
| Motion coverage showed 138% | overlapping events on two stems were summed | Overlapping time is counted once. |
| BEATs reloaded for every stem | no caching | Loaded once and reused. |
| Results lost after a server restart | job list only in memory | Results and downloads fall back to the files on disk. |
| 2-stem separator crashed on audio shorter than 16 s | the model returns slightly fewer samples than it is given | The output is padded to the input length. |

### Data collection

| Problem | Cause | Fix |
|---|---|---|
| FreeSounds `unauthorized_client` | the OAuth grant is unavailable on free accounts | Pass the client secret as the `token=` query parameter. |
| `pip install freesound` fails | no such package | Use the REST API with `requests`. |
| yt-dlp downloaded 0 clips | ffmpeg DLL conflict in the conda env | `static_ffmpeg.add_paths()`. |
| yt-dlp downloaded few clips | a strict duration filter silently rejected most | Filter removed, more results per query. |
| Slakh "first 700 tracks" gave incomplete tracks | files in the archive are not grouped by track | Download the whole archive (resumable), extract stems only, delete the archive. |
| FreeSounds key in git | hardcoded in `collect_stems.py` | Moved to `backend\.env`. **The old key is still in git history and must be rotated.** |
| MelBandRoformer loads with 842 missing weights; hybrid stems do not add up to the song | the checkpoint does not match the architecture (this was a suspected issue in June, confirmed 18 Nov) | Default changed to pure mode; hybrid kept as an option, not recommended. |
| Windows blocked PyTorch (`Application Control policy has blocked ... caffe2_nvrtc.dll`), then briefly the project folder (access denied) | a Windows security policy; cleared by the owner and a restart | Nothing in the project changed; the owner allowed it in Windows Security. |
| Real songs landed mostly in the Folk / Texture stem | that category's clips were whole folk songs, found by searches such as "Telugu folk music" | `clean_folk_texture.py` keeps only clips the detector confirms as percussion without voice or melody. |

### Environment and tooling

| Problem | Cause | Fix |
|---|---|---|
| NumPy silently upgraded to 2.x (at least 3 times) | other pip installs | `pip install numpy==1.26.4 --force-reinstall`; check after every install. |
| "PyTorch >= 2.4 is required" message | the `transformers` package | Harmless. Ignore. |
| ffmpeg popup about `gdk_pixbuf` | conda/GTK DLL conflict | Cosmetic. Click OK. |
| Git push rejected, huge repo | venv, models and outputs were committed | History rewritten; `.gitignore`. |
| "Site can't be reached" for the UI | the dev server is on port 3000, not 5173 | Open `localhost:3000`. |
| Edits "not taking effect" | files copied into the wrong folder | One project folder; launch scripts; every path from `paths.py`. |
| `dir | find "File(s)"` reported 0 mixes | the folder contained only sub-folders | Count folders, not files. |
| Mixed line endings, git warnings | files written by different tools | Normalised; `.gitattributes`. |
| `requirements.txt` did not match reality | never updated | Rewritten from the installed versions. |

---

## 13. Open problems

| # | Problem | Detail |
|---|---|---|
| 1 | **The model is good, not great** | Final: random 3.5 dB, coherent 6.6 dB. Keys 1.5 dB, plucked strings 3.1, folk 2.7 on random mixes. Gains over the last 60 epochs were small, so more epochs will not help much; better data or a bigger model would. |
| 2 | Folk / texture swallowing real songs | **Fixed**: 4.2% of a real song at epoch 93, down from 64% ([section 14](#14-the-folk--texture-problem)). |
| 3 | Nobody has listened yet | Every quality claim in this file is a measurement. The stems need human ears. |
| 4 | Hybrid mode is unreliable | Tested 18 Nov: MelBandRoformer loads with 842 missing weights and the stems no longer sum to the song. Off by default. Fixing it would need the correct checkpoint or config. |
| 5 | Slakh conversion unfinished | 7,308 stems left; about 1 to 2 hours alone. |
| 6 | Tier 1 data is noisy | YouTube "solos" with accompaniment, lossy previews. |
| 7 | No clean data for several named instruments | veena, nadaswaram, thavil, bansuri, harmonium, dappu, parai. |
| 8 | BEATs does not know Indian instruments | It names "Guitar" or "Drum machine"; it cannot say "veena" or "mridangam". (The raw-code gaps were filled on 22 Nov.) |
| 9 | Model works at 16 kHz mono | Stems are full-band only through the mask trick in [section 4](#4-how-the-app-works). Above 7.5 kHz the split is approximate. Input is mono; the original stereo image is not used. |
| 10 | Licences | Three datasets are non-commercial. |
| 11 | FreeSounds secret | Still needs rotating on the FreeSounds site. |
| 12 | Three phase tests not re-run | `test_separator.py`, `test_detector.py`, `test_renderer.py` import correctly after the reorganisation but were not executed. |
| 13 | Validation includes noisy Tier 1 recordings | The overall median therefore understates quality on clean data. |

---

## 14. The Folk / Texture problem

**State: the data was cleaned on 13 November 2026. The effect on the model
is not yet measured.**

### What happened

When the epoch 2 model splits a real song, most of the song lands in the
Folk / Texture stem.

| Song | Folk / texture | Bass | Vocals | All five other stems together |
|---|---|---|---|---|
| `test5.mp3` (5 min) | **64%** | 21% | 7.5% | 8% |
| `tests\test_song.flac` (30 s) | **63%** | 2% | 22% | 13% |

So the stem that should hold dappu, parai and dhol instead holds the bulk of
the music. Everything placed in it ends up far left at the back (layer 8),
and the other stems are left nearly empty.

### Why

The model learned what it was shown. The Folk / Texture training clips are
not isolated instruments. They are pieces of **whole folk songs**.

1. **The search terms asked for songs.** `download_youtube.py` searched for
   "Telugu folk music", "Telangana folk song", "South Indian folk music",
   "Tamil folk music", "Andhra folk music", "village music India", "folk
   dance music India", "Bonalu festival music". These return complete
   performances with singers and several instruments.
2. **The recordings are long.** The 84 source recordings give 8,163 clips;
   the median recording gives 48 clips (about 4 minutes), and the longest
   gives 912 (76 minutes). Ten recordings supply more than half the clips.
   For comparison, the median percussion recording gives 4 clips.
3. **The detector hears voices in them.** BEATs was run on one clip from each
   of 48 folk recordings, and on 48 percussion recordings for comparison:

   | | Folk / texture clips | Percussion clips |
   |---|---|---|
   | contain a voice (singing or speech) | 29% | 8% |
   | contain detectable percussion | 40% | 77% |

4. **No clean data balances it.** Folk / texture is the only category with
   no Tier 2 clips at all (0 of 68,637).

During training, a Folk / Texture "stem" is therefore a full mix: voices,
drums and melody together. The model's lesson is: **anything that sounds
like a complete piece of music is Folk / Texture.** A real film song sounds
exactly like that, so it goes there.

A second, smaller problem sits underneath: even with perfect data, dappu,
parai and dhol are drums. Asking a model to tell "folk drum" from "drum" is
asking it to split one kind of sound into two stems.

### The options that were considered

| Option | What changes | Cost | Risk |
|---|---|---|---|
| **A. Clean the folk clips (chosen)** | Keep only folk clips where the detector hears percussion and no voice or melody. Stem 8 becomes "folk percussion". | A script and an hour. Training resumes from epoch 3. | Folk and percussion stems will blur into each other, because they are the same kind of sound. |
| B. Drop the stem | 7 stems. Dappu, parai and dhol go to Percussion. | Restart training from zero. Small changes to the layout and UI. | Loses the separate folk zone that was part of the original design. |
| C. Make it an honest "everything else" stem | Rename stem 8 and feed it only sounds that belong in no other category. | Needs data that does not exist yet. | Without data it stays a dumping ground. |
| D. Leave it | Nothing. | None. | The problem above, trained in harder. |

### What was done (13 November 2026)

The project owner chose option A. `backend\data_tools\clean_folk_texture.py`
ran BEATs over all 8,163 clips and sorted them:

| Group | Clips | Share | Decision |
|---|---|---|---|
| Percussion, no voice, no melody | 1,612 | 20% | **kept** |
| Percussion with a voice or a melodic instrument | 1,057 | 13% | removed |
| Voice or melody, no percussion | 2,187 | 27% | removed |
| Nothing musical recognised (generic "Music", ambient and animal sounds) | 2,876 | 35% | removed |
| Weak percussion only | 431 | 5% | removed |

A sample of the kept clips was checked: the detector hears "Percussion" in
100% of them and "Drum" in 93%.

The rule: a clip is kept when its percussion score is at least 0.20, its
voice score at most 0.10 and its melodic score at most 0.15. A score is the
highest BEATs probability for that family in any 4-second window of the clip.
The thresholds are constants at the top of the script.

Result: Folk / texture now has **1,612 clips from 48 recordings** (was 8,163
from 84). Training uses 1,437 clips from 44 of those recordings; 4 recordings
are held out for validation.

**Nothing was deleted.** The 6,551 removed clips are in
`data\processed\rejected\folk_texture\`. To undo:

```cmd
python data_tools\clean_folk_texture.py --restore
```

To re-apply after changing the thresholds: `python data_tools\clean_folk_texture.py --apply`.
The scores are saved in `data\processed\folk_texture_scores.json`, so reruns
take seconds.

### What is not known yet

- **Whether it worked.** The epoch 3 checkpoint was trained on the old clips
  and still puts songs into Folk / Texture. It has to be trained further on
  the cleaned data. Check after about 10 epochs with
  `python training\evaluate_separator.py`: Folk / Texture should hold a small
  share of a real song. If it still holds a large share, switch to option B.
- **Scores before and after are not directly comparable.** The validation set
  includes held-out folk recordings, and three of the seven were removed
  entirely. Epoch 4 onward is scored on a slightly different set from
  epochs 1 to 3.
- **Folk and percussion will overlap.** A dappu and a drum kit are the same
  kind of sound. Expect the two stems to share material.
- **Some good clips were probably lost.** BEATs does not know dappu or parai
  by name. Clips where it recognised nothing (35%) may include real folk
  instruments it could not identify.
- **The other Tier 1 categories were not checked.** A sample of Tier 1
  percussion clips had a voice in 8% of them. The same tool could be adapted.

---

## 15. What is left to do

In order:

1. ~~Decide the Folk / Texture question~~ Done on 13 November: the clips
   were cleaned ([section 14](#14-the-folk--texture-problem)). What remains is
   to confirm it after about 10 epochs of training.
2. **Finish preparing the data and train**: double-click
   `start_data_then_training.bat`. It runs three steps in order: the Slakh
   conversion (about 47 minutes), the coherent cutting (about 2 hours,
   estimated), then training from zero (about 30 minutes per epoch alone, so
   about 2 days for 100 epochs). 100 is a ceiling; check the scores every 10
   epochs.
3. **Watch the coherent score.** It is the number closest to real-song
   quality. If it stays far below the random score, the model is still
   learning the easy problem.
4. **Listen.** The 8 stems of `test5.mp3` (epoch 93) are in
   `data\evaluation\test5`. Check plucked strings, wind and bass in
   particular.
5. ~~Run a real song through the app in hybrid mode~~ Done 18 Nov: hybrid mode
   is unreliable and was switched off (section 4).
6. **Rotate the FreeSounds secret.**
7. **Name Indian instruments**: fine-tune BEATs on Indian instrument labels,
   or add a small classifier per stem.
8. **Reference songs.** The owner wants to supply "Ordinary Person" (Leo) and
   "Infatuation" (100% Love), have their bass character and 360° layering
   analysed, and then choose which characteristics to train. Not started; the
   songs have not been provided.
9. **More data** for veena, nadaswaram, thavil, bansuri, harmonium and folk
   percussion.

### Rules set by the project owner

- No Demucs. The Western 4 and 6 stem models are not acceptable.
- All kinds of instruments must be detected and placed, especially South
  Indian ones.
- Each stem needs its own direction and its own depth layer.
- For reference-song training, the songs are analysed first and the owner
  decides which characteristics get trained. Nothing is decided in advance.

---

## 15a. v8 and the 60-song benchmark (25 Nov 2026)

v8 (9 stems, started from v6, five epochs, SDR 3.2 dB) scored 80% recall on the 11-song scorecard, level with v5, with 4 false flags per song set against 9 for v5. One song (Unakku Thaan) lost 25 points, so the promotion rule kept v5 as the app model.

The 60-song benchmark (random 60 s excerpts from a personal folder, truth from BEATs tags) gave 100% recall for v5, v6, v7 and v8, and 0.62-0.72 flags per song. The models cannot be told apart on it, so nothing changed. Vocal dominance and the bass sink remain open.

## 16. Environment

| Item | Value |
|---|---|
| Machine | Windows 11 laptop, NVIDIA RTX 3060 Laptop GPU (6 GB), 24 GB RAM |
| Python | 3.11, conda env `raga-spatial` in `C:\ProgramData\miniconda3\envs\raga-spatial` |
| PyTorch | **2.3.0+cu118 (locked)** |
| NumPy | **1.26.4 (locked)** |
| Other packages | `backend\requirements.txt` |
| Frontend | Node 24, Vite 8, React 19, axios. Dev server on port 3000, proxies `/api` to port 8000. |
| Git | `github.com/karthikrachala1507/raga-spatial`, branch `main` |
| Disk | project on D: (618 GB free); C: has about 33 GB free |

The conda environment stays on C:. Conda environments cannot simply be moved.

After any `pip install`, check the locked versions:

```cmd
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

Expect `1.26.4 2.3.0+cu118`.

The FreeSounds key lives in `backend\.env` (copy `backend\.env.example`).
Only `collect_stems.py` needs it.

---

## 17. Notes for AI assistants

Things that cost time and are not obvious from the code.

- **Start here:** this file, then `backend\paths.py`, then
  `backend\core\renderer.py` (the pipeline) and
  `backend\training\train_separator.py` (training).
- **Run scripts from `backend\`.** Tools and tests add `backend\` to the
  import path themselves, but relative paths in the old phase tests assume it.
- **Activate the conda env, or put its folders on PATH.** Calling
  `python.exe` directly without activation fails at the first GPU complex
  operation with `nvrtc: error: failed to open nvrtc-builtins64_118.dll`,
  because that DLL lives in the env's `bin` folder.
- **In a sandbox that cannot write to the conda folder,** librosa hangs
  forever on first use while numba retries writing its cache. Set
  `NUMBA_CACHE_DIR` to a writable folder.
- **`CUDA_VISIBLE_DEVICES=""` does not hide the GPU on Windows.** Use
  `RAGA_DEVICE=cpu` for the separator.
- **Do not run a second GPU job while training runs** without checking
  memory. Training uses about 4.4 GB of 6.4 GB.
- **Do not edit `train_separator.py` while training is running.** Its data
  workers re-import the file at every epoch.
- **Do not trust the loss alone.** v1's loss fell for 100 epochs while the
  output went silent. Use `evaluate_validation.py` (dB) and listen.
- **Checkpoints hold weights and the model's settings only**, so the model
  class can be moved between files without breaking them.
- **Windows shell traps:** `sed` and heredocs eat backslashes (`\t` becomes a
  tab); the console is cp1252 and cannot print box-drawing characters; `du`
  and recursive listings over the data folder take minutes.
- **The three `.json` files under `data\logs` are the only record of past
  training.** Do not delete them.
- **Never commit:** `data\`, `backend\models\`, `backend\outputs\`,
  `backend\.env`, `docs\`.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| Stem | One separated part of a song, such as vocals or bass. |
| Separation | Splitting a mixed song into stems. Different from detection. |
| Detection | Naming which instruments are present. Does not split anything. |
| Binaural | Two-channel audio made for headphones that gives a sense of direction. |
| HRTF | Head-related transfer function: how the head and ears colour a sound from a given direction. Convolving a sound with it places the sound there. |
| Azimuth | Direction around the listener in degrees. 0° is front. |
| SDR | Signal-to-distortion ratio in dB. How close a separated stem is to the true one. 0 dB means no better than silence; higher is better. |
| Epoch | One round of training. Here, 12,000 freshly built mixes. |
| Checkpoint | A saved copy of the model during training. |
| Held-out | Recordings kept out of training so the score reflects new material. |
| Mask | A per-frequency, per-moment share that says how much of the mix belongs to a stem. |
| Bleed | Sound from one instrument leaking into another instrument's stem. |
| Tier 1 / Tier 2 | The first data (FreeSounds, YouTube) / the open datasets added in November. |
| BSRNN | Band-split recurrent neural network, the design of the 8-stem model. |
| Mixed precision | Running parts of the model in 16-bit floats to save GPU memory. |
