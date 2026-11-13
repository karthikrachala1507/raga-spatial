# Raga Spatial

Spatial (binaural) audio for South Indian film music: upload a song, get it
split into 8 instrument stems, each placed at its own direction and depth
around the listener.

Everything for this project lives in this one folder.

## Layout

```
raga-spatial\
├── start_app.bat              double-click: starts the app and opens the browser
├── start_training.bat         double-click: start or resume training
├── README.md  DATASETS.md
│
├── backend\
│   ├── paths.py               the one file that says where everything lives
│   ├── main.py  api\          the web server (FastAPI) and its routes
│   ├── core\                  the audio pipeline
│   │   ├── preprocess.py        load audio, tempo, onsets, chroma
│   │   ├── separator_2stem.py   MelBandRoformer: vocals + instrumental
│   │   ├── separator_8stem.py   the custom model: 8 full-bandwidth stems
│   │   ├── bsrnn.py             the 8-stem model itself
│   │   ├── detector.py          BEATs: names the instruments in each stem
│   │   ├── motion.py            finds melodic runs that should move in space
│   │   ├── spatial.py           positions by instrument label (2-stem path)
│   │   ├── spatial_8stem.py     the 8 positions, depth layers and rendering
│   │   ├── hrtf.py              binaural (HRTF) rendering and distance effects
│   │   └── renderer.py          runs the whole pipeline for one song
│   ├── training\
│   │   ├── train_separator.py     trains the 8-stem model
│   │   ├── evaluate_validation.py scores a checkpoint in dB
│   │   └── evaluate_separator.py  writes the 8 stems of a real song, to listen to
│   ├── data_tools\            download and prepare training data
│   ├── tests\                 checks and diagnostics (test_song.flac is the test song)
│   ├── models\                MelBandRoformer, BEATs and HRTF files used by the app
│   └── outputs\               what the app produces for each uploaded song
│
├── frontend\                  the React UI (8-zone map and stem mixer)
│
└── data\                      everything large (not in git)
    ├── raw_stems\             Tier 1 raw downloads (FreeSounds, YouTube)
    ├── datasets\              open datasets, stems only
    ├── processed\stems_t1\    Tier 1 training clips
    ├── processed\stems_t2\    Tier 2 training clips (from the open datasets)
    ├── checkpoints\           trained separator (separator_v3_best.pt is used by the app)
    ├── logs\                  training_log_v3.json, one entry per epoch
    ├── real_songs\            full songs for listening tests
    └── evaluation\            stems written by evaluate_separator.py
```

There are two model folders on purpose: `backend\models\` holds the
downloaded models the app needs, `data\checkpoints\` holds what training
produces.

The Python environment is the conda env `raga-spatial` in
`C:\ProgramData\miniconda3\envs\raga-spatial`. It stays there: conda
environments cannot simply be moved. PyTorch 2.3.0+cu118 and NumPy 1.26.4
are locked; check after any `pip install` (see `backend\requirements.txt`).

## Everyday commands

The two things you do most need no terminal:

| What | How |
|---|---|
| Start the app | double-click `start_app.bat` |
| Start or resume training | double-click `start_training.bat` |
| Pause training | Ctrl+C in its window (it resumes from the last finished epoch) |

For everything else, open a terminal and go to the backend folder first:

```cmd
conda activate raga-spatial
cd /d D:\raga-spatial\backend
```

| What | Command |
|---|---|
| Score the current model | `python training\evaluate_validation.py` |
| Write stems of a real song | `python training\evaluate_separator.py --song "D:\path\song.mp3"` |
| Download the open datasets | `python data_tools\download_datasets.py` |
| Turn datasets into clips | `python data_tools\prepare_datasets.py` |
| Quick audio checks | `python tests\test_8stem.py` |
| Whole pipeline on the test song | `python tests\test_8stem.py pipeline` |

`download_datasets.py` and `prepare_datasets.py` can be stopped and rerun at
any time; they skip what is already done. Training picks up new clips at the
start of each epoch.

## Reading the training output

Each epoch prints `SDR: x dB` and a line of per-stem numbers. This is the
separation quality on songs the model never trained on:

| SDR | Meaning |
|---|---|
| 0 dB | no better than silence |
| 1-3 dB | starting to separate, heavy bleed between stems |
| 4-6 dB | usable for spatial placement |
| 7+ dB | good |

## The 8 stems

| Layer | Stem | Direction | Depth |
|---|---|---|---|
| 1 | Vocals | front | 0.20 |
| 2 | Plucked strings (veena, sitar, guitar) | front-left | 0.35 |
| 3 | Wind (bansuri, nadaswaram, flute, brass) | front-right | 0.45 |
| 4 | Keys / synth (piano, harmonium) | front-right | 0.50 |
| 5 | Bowed strings (violin, cello) | rear-left | 0.60 |
| 6 | Percussion (tabla, mridangam, drums) | rear | 0.65 |
| 7 | Bass | rear-right | 0.75 |
| 8 | Folk / texture (dappu, parai, dhol) | left | 0.85 |

Positions and depths are set in `backend\core\spatial_8stem.py` (`STEM_LAYOUT`).
