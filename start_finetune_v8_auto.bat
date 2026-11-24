@echo off
rem Raga Spatial - v8 fine-tune, reference-song analysis, scorecard and automatic promotion decision.
rem Starts from v6 (best scorecard so far), 9 stems, stronger balance settings.
call "C:\ProgramData\miniconda3\condabin\conda.bat" activate raga-spatial
set "LOG=%~dp0data\logs\pipeline_progress.log"
set NUMBA_CACHE_DIR=%~dp0data\numba_cache
set PYTHONIOENCODING=utf-8
cd /d "%~dp0backend"
echo %date% %time% START v8 finetune>> "%LOG%"
set RAGA_RUN_NAME=separator_v8
set RAGA_INIT_FROM=separator_v6_best.pt
set RAGA_EPOCHS=5
set RAGA_LR=0.0002
set RAGA_LR_DECAY=0.85
set RAGA_COHERENT_PROB=0.8
set RAGA_MASTER_PROB=0.8
set RAGA_ORIGIN_BALANCE=0.3
set RAGA_SAVE_EVERY=1
set RAGA_LOSS_NORM=0.7
set RAGA_ABSENT_W=0.4
set RAGA_DROP_PROB=0.3
python training\train_separator.py
set RAGA_RUN_NAME=
set RAGA_INIT_FROM=
if not exist "%~dp0data\checkpoints\separator_v8_best.pt" goto stopped
echo %date% %time% START v8 reference songs>> "%LOG%"
set RAGA_SEP8_MODEL=%~dp0data\checkpoints\separator_v8_best.pt
set RAGA_REF_TAG=_v8
python training\analyze_reference_songs.py
set RAGA_SEP8_MODEL=
set RAGA_REF_TAG=
python training\scorecard_reference_songs.py > "%~dp0data\logs\scorecard_run.txt" 2>&1
python training\auto_promote.py v8 v5 >> "%~dp0data\logs\auto_promote.txt" 2>&1
echo %date% %time% DONE v8 finetune scorecard promote>> "%LOG%"
exit /b 0
:stopped
echo %date% %time% STOPPED v8 before finishing>> "%LOG%"
exit /b 1
