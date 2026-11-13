# Raga Spatial - paths.py
# The ONE place that says where things live. Every script imports from here,
# so the whole project can be moved by moving its folder.
#
#   <project>\                    e.g. D:\raga-spatial
#   ├── backend\                  the app, training, tools and tests (this folder)
#   ├── frontend\                 the React UI
#   └── data\                     everything large: datasets, clips, checkpoints, logs
#
# To keep the data somewhere else, set the environment variable RAGA_DATA_DIR.

import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
DATA_DIR    = os.environ.get("RAGA_DATA_DIR", os.path.join(PROJECT_DIR, "data"))

RAW_STEMS_DIR   = os.path.join(DATA_DIR, "raw_stems")              # Tier 1 raw downloads
STEMS_T1_DIR    = os.path.join(DATA_DIR, "processed", "stems_t1")  # Tier 1 clips (FreeSounds, YouTube)
STEMS_T2_DIR    = os.path.join(DATA_DIR, "processed", "stems_t2")  # Tier 2 clips (open datasets)
STEMS_ALIGNED_DIR = os.path.join(DATA_DIR, "processed", "stems_aligned")  # same-song, same-moment stems (coherent mixes)
REJECTED_DIR    = os.path.join(DATA_DIR, "processed", "rejected")  # clips removed from training (kept, not deleted)
DATASETS_DIR    = os.path.join(DATA_DIR, "datasets")               # open datasets, stems only
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")            # separator checkpoints from training
LOGS_DIR        = os.path.join(DATA_DIR, "logs")                   # training logs
REAL_SONGS_DIR  = os.path.join(DATA_DIR, "real_songs")             # full songs for listening tests
EVAL_DIR        = os.path.join(DATA_DIR, "evaluation")             # stems written by evaluate_separator.py

APP_MODELS_DIR  = os.path.join(BACKEND_DIR, "models")              # MelBandRoformer, BEATs, HRTF file
OUTPUTS_DIR     = os.path.join(BACKEND_DIR, "outputs")             # what the app produces per job
