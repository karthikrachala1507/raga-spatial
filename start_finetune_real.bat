@echo off
rem Raga Spatial - fine-tune v5 on real-sounding mixes, then score it on the 11 reference songs.
rem
rem   Step 1  fine-tune from separator_v3_best.pt, 6 epochs (about 3 hours)
rem           80% coherent mixes (real multitrack windows), 80% of mixes get a
rem           mastering chain (EQ, reverb, bus compression), fewer synthetic mixes
rem           -> data\checkpoints\separator_v5_*.pt (one file per epoch), training_log_v5.json
rem   Step 2  run v5 on the 11 reference songs (about 1 hour)
rem   Step 3  write the scorecard: v3 and v4 against v5 -> data\evaluation\reference_songs\SCORECARD.md
rem
rem separator_v3_best.pt is never overwritten. Pause with Ctrl+C (answer Y); run this
rem file again to continue (the fine-tune resumes from its last epoch).
call "C:\ProgramData\miniconda3\condabin\conda.bat" activate raga-spatial
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
set "LOG=%~dp0data\logs\pipeline_progress.log"
set NUMBA_CACHE_DIR=%~dp0data\numba_cache
set PYTHONIOENCODING=utf-8
cd /d "%~dp0backend"

echo ===== Step 1 of 3: fine-tune v5 =====
echo %date% %time% START real-data step 1 finetune v5>> "%LOG%"
set RAGA_RUN_NAME=separator_v5
set RAGA_INIT_FROM=separator_v3_best.pt
set RAGA_EPOCHS=6
set RAGA_LR=0.0003
set RAGA_LR_DECAY=0.85
set RAGA_COHERENT_PROB=0.8
set RAGA_MASTER_PROB=0.8
set RAGA_ORIGIN_BALANCE=0.3
set RAGA_SAVE_EVERY=1
python training\train_separator.py
set RAGA_RUN_NAME=
set RAGA_INIT_FROM=
set RAGA_EPOCHS=
set RAGA_LR=
set RAGA_LR_DECAY=
set RAGA_COHERENT_PROB=
set RAGA_MASTER_PROB=
set RAGA_ORIGIN_BALANCE=
set RAGA_SAVE_EVERY=
if not exist "%~dp0data\checkpoints\separator_v5_best.pt" goto stopped

echo ===== Step 2 of 3: v5 on the reference songs =====
echo %date% %time% START real-data step 2 reference songs>> "%LOG%"
set RAGA_SEP8_MODEL=%~dp0data\checkpoints\separator_v5_best.pt
set RAGA_REF_TAG=_v5
python training\analyze_reference_songs.py
set RAGA_SEP8_MODEL=
set RAGA_REF_TAG=

echo ===== Step 3 of 3: scorecard =====
echo %date% %time% START real-data step 3 scorecard>> "%LOG%"
python training\scorecard_reference_songs.py > "%~dp0data\logs\scorecard_run.txt" 2>&1
echo %date% %time% DONE real-data fine-tune and scorecard>> "%LOG%"
echo.
echo DONE. Scorecard: %~dp0data\evaluation\reference_songs\SCORECARD.md
exit /b 0

:stopped
echo %date% %time% STOPPED real-data run before finishing>> "%LOG%"
echo Stopped before finishing. Run this file again to continue.
exit /b 1
