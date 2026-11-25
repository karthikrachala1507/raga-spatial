@echo off
rem Raga Spatial - sub-bass fine-tune, then an automatic before/after comparison.
rem
rem   Step 1  generate 500 synthetic 808-style bass tracks   (about 3 minutes, skipped if done)
rem   Step 2  fine-tune from separator_v3_best.pt, 5 epochs   (about 2.5 hours)
rem           -> data\checkpoints\separator_v4_*.pt, data\logs\training_log_v4.json
rem   Step 3  score the old and the new model on the SAME validation set
rem   Step 4  re-analyse "Ordinary Person" with the new model
rem
rem The results are written to data\logs\finetune_subbass_report.txt.
rem separator_v3_best.pt is never overwritten. Pause with Ctrl+C (answer Y); run
rem this file again to continue (the fine-tune resumes from its last epoch).
call "%~dp0env.bat"
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
set "LOG=%~dp0data\logs\pipeline_progress.log"
set "REPORT=%~dp0data\logs\finetune_subbass_report.txt"
cd /d "%~dp0backend"

echo ===== Step 1 of 4: synthetic sub-bass =====
echo %date% %time% START subbass step 1 generate>> "%LOG%"
python data_tools\make_synthetic_bass.py 500
if errorlevel 1 goto stopped

echo ===== Step 2 of 4: fine-tune =====
echo %date% %time% START subbass step 2 finetune>> "%LOG%"
set RAGA_RUN_NAME=separator_v4
set RAGA_INIT_FROM=separator_v3_best.pt
set RAGA_EPOCHS=5
set RAGA_LR=0.0002
set RAGA_LR_DECAY=0.9
python training\train_separator.py
set RAGA_RUN_NAME=
set RAGA_INIT_FROM=
set RAGA_EPOCHS=
set RAGA_LR=
set RAGA_LR_DECAY=
if not exist "%~dp0data\checkpoints\separator_v4_best.pt" goto stopped

echo ===== Step 3 of 4: compare old and new model =====
echo %date% %time% START subbass step 3 compare>> "%LOG%"
echo OLD MODEL (separator_v3_best.pt) on the current validation set> "%REPORT%"
python training\evaluate_validation.py separator_v3_best.pt 120 >> "%REPORT%" 2>nul
echo.>> "%REPORT%"
echo NEW MODEL (separator_v4_best.pt) on the same validation set>> "%REPORT%"
python training\evaluate_validation.py separator_v4_best.pt 120 >> "%REPORT%" 2>nul

echo ===== Step 4 of 4: Ordinary Person with the new model =====
echo %date% %time% START subbass step 4 reference song>> "%LOG%"
set "REFDIR=%~dp0data\evaluation\reference_songs"
if exist "%REFDIR%\Ordinary_Person.json" move /y "%REFDIR%\Ordinary_Person.json" "%REFDIR%\Ordinary_Person_v3.json" >nul
set RAGA_SEP8_MODEL=%~dp0data\checkpoints\separator_v4_best.pt
python training\analyze_reference_songs.py "Ordinary Person"
set RAGA_SEP8_MODEL=
if exist "%REFDIR%\Ordinary_Person.json" move /y "%REFDIR%\Ordinary_Person.json" "%REFDIR%\Ordinary_Person_v4.json" >nul
echo %date% %time% DONE sub-bass fine-tune and comparison>> "%LOG%"
echo.
echo DONE. Report: %REPORT%
exit /b 0

:stopped
echo %date% %time% STOPPED sub-bass run before finishing>> "%LOG%"
echo Stopped before finishing. Run this file again to continue.
exit /b 1
