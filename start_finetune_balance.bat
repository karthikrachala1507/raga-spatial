@echo off
rem Raga Spatial - fine-tune v6 to stop vocals and bass swallowing other stems, then score it on the 11 reference songs.
rem
rem   Step 1  fine-tune v6, 6 epochs (about 3 hours)
rem           starts from separator_v5_best.pt; level-normalised loss (quiet stems count),
rem           stronger penalty on absent stems, vocals/bass dropped from 30% of real windows
rem           -> data\checkpoints\separator_v6_*.pt (one file per epoch), training_log_v6.json
rem   Step 2  run v6 on the 11 reference songs (about 1 hour)
rem   Step 3  write the scorecard: v3, v4 and v5 against v6 -> data\evaluation\reference_songs\SCORECARD.md
rem
rem Earlier models are never overwritten. Pause with Ctrl+C (answer Y); run this
rem file again to continue (the fine-tune resumes from its last epoch).
call "%~dp0env.bat"
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
set "LOG=%~dp0data\logs\pipeline_progress.log"
set NUMBA_CACHE_DIR=%~dp0data\numba_cache
set PYTHONIOENCODING=utf-8
cd /d "%~dp0backend"

echo ===== Step 1 of 3: fine-tune v6 =====
echo %date% %time% START balance step 1 finetune v6>> "%LOG%"
set RAGA_RUN_NAME=separator_v6
set RAGA_INIT_FROM=separator_v5_best.pt
set RAGA_EPOCHS=6
set RAGA_LR=0.0002
set RAGA_LR_DECAY=0.85
set RAGA_COHERENT_PROB=0.8
set RAGA_MASTER_PROB=0.8
set RAGA_ORIGIN_BALANCE=0.3
set RAGA_SAVE_EVERY=1
set RAGA_LOSS_NORM=0.7
set RAGA_ABSENT_W=0.3
set RAGA_DROP_PROB=0.3
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
set RAGA_LOSS_NORM=
set RAGA_ABSENT_W=
set RAGA_DROP_PROB=
if not exist "%~dp0data\checkpoints\separator_v6_best.pt" goto stopped

echo ===== Step 2 of 3: v6 on the reference songs =====
echo %date% %time% START balance step 2 reference songs>> "%LOG%"
set RAGA_SEP8_MODEL=%~dp0data\checkpoints\separator_v6_best.pt
set RAGA_REF_TAG=_v6
python training\analyze_reference_songs.py
set RAGA_SEP8_MODEL=
set RAGA_REF_TAG=

echo ===== Step 3 of 3: scorecard =====
echo %date% %time% START balance step 3 scorecard>> "%LOG%"
python training\scorecard_reference_songs.py > "%~dp0data\logs\scorecard_run.txt" 2>&1
echo %date% %time% DONE balance fine-tune and scorecard>> "%LOG%"
echo.
echo DONE. Scorecard: %~dp0data\evaluation\reference_songs\SCORECARD.md
exit /b 0

:stopped
echo %date% %time% STOPPED balance run before finishing>> "%LOG%"
echo Stopped before finishing. Run this file again to continue.
exit /b 1
