@echo off
rem Raga Spatial - finish preparing ALL the training data, THEN train.
rem The jobs run ONE AT A TIME, shortest first, because they all read heavily
rem from the same disk and slow each other down when run together.
rem
rem   Step 1  recover short raw clips                about 2 minutes
rem   Step 2  single clips: finish Slakh             about 5-20 minutes
rem   Step 3  coherent stems: Slakh                  about 70 minutes
rem   Step 4  download Saraga Audiovisual stems      about 2 hours (network)
rem   Step 5  single clips: Saraga Audiovisual       about 20-30 minutes
rem   Step 6  coherent stems: Saraga Audiovisual     about 20-30 minutes
rem   Step 7  training                               about 30 minutes per epoch
rem
rem Every step skips what is already done. Pause with Ctrl+C (answer Y) and run
rem this file again to continue. Step times are written to
rem   data\logs\pipeline_progress.log
call "%~dp0env.bat"
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
set "LOG=%~dp0data\logs\pipeline_progress.log"
cd /d "%~dp0backend"
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo ===== Step 1 of 7: recover short raw clips =====
echo %date% %time% START step 1 recover short clips>> "%LOG%"
python data_tools\recover_short_clips.py
if errorlevel 1 goto stopped

echo ===== Step 2 of 7: finish Slakh single clips =====
echo %date% %time% START step 2 slakh clips>> "%LOG%"
python data_tools\prepare_datasets.py saraga_carnatic musdb18hq urmp guitarset mridangam_tani tabla_4way medley_solos slakh2100
if errorlevel 1 goto stopped

echo ===== Step 3 of 7: Slakh coherent stems =====
echo %date% %time% START step 3 slakh coherent>> "%LOG%"
python data_tools\prepare_coherent.py slakh2100 urmp musdb18hq saraga_carnatic
if errorlevel 1 goto stopped

echo ===== Step 4 of 7: download Saraga Audiovisual stems =====
echo %date% %time% START step 4 saraga audiovisual download>> "%LOG%"
python data_tools\download_datasets.py saraga_audiovisual
if errorlevel 1 goto stopped
if not exist "%~dp0data\datasets\saraga_audiovisual\.done" goto stopped

echo ===== Step 5 of 7: Saraga Audiovisual single clips =====
echo %date% %time% START step 5 saraga audiovisual clips>> "%LOG%"
python data_tools\prepare_datasets.py saraga_audiovisual
if errorlevel 1 goto stopped

echo ===== Step 6 of 7: Saraga Audiovisual coherent stems =====
echo %date% %time% START step 6 saraga audiovisual coherent>> "%LOG%"
python data_tools\prepare_coherent.py saraga_audiovisual
if errorlevel 1 goto stopped

echo ===== Step 7 of 7: training =====
echo %date% %time% START step 7 training>> "%LOG%"
python training\train_separator.py
echo %date% %time% END step 7 training (finished or paused)>> "%LOG%"
exit /b 0

:stopped
echo %date% %time% STOPPED before finishing>> "%LOG%"
echo.
echo Data preparation stopped before finishing. Training was NOT started.
echo Run this file again to continue.
exit /b 1
