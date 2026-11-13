@echo off
rem Raga Spatial - finish preparing the training data, THEN train.
rem The jobs run ONE AT A TIME, shortest first. They all read heavily from the
rem same disk, so running them together makes each about three times slower.
rem
rem   Step 1  coherent stems: MUSDB18-HQ        about 10 minutes
rem   Step 2  coherent stems: Saraga Carnatic   about 30-45 minutes
rem   Step 3  single clips: finish Slakh        about 47 minutes
rem   Step 4  coherent stems: Slakh             about 70 minutes
rem   Step 5  training                          about 30 minutes per epoch
rem
rem Every step skips what is already done. Pause at any time with Ctrl+C
rem (answer Y) and run this file again to continue.
rem Start and finish times of each step are written to
rem   data\logs\pipeline_progress.log
call "C:\ProgramData\miniconda3\condabin\conda.bat" activate raga-spatial
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
set "LOG=%~dp0data\logs\pipeline_progress.log"
cd /d "%~dp0backend"
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo.
echo ===== Step 1 of 5: coherent stems, MUSDB18-HQ =====
echo %date% %time% START step 1 musdb coherent>> "%LOG%"
python data_tools\prepare_coherent.py musdb18hq
if errorlevel 1 goto stopped
echo %date% %time% DONE step 1>> "%LOG%"

echo.
echo ===== Step 2 of 5: coherent stems, Saraga Carnatic =====
echo %date% %time% START step 2 saraga coherent>> "%LOG%"
python data_tools\prepare_coherent.py saraga_carnatic
if errorlevel 1 goto stopped
echo %date% %time% DONE step 2>> "%LOG%"

echo.
echo ===== Step 3 of 5: single clips, finish Slakh =====
echo %date% %time% START step 3 slakh clips>> "%LOG%"
python data_tools\prepare_datasets.py
if errorlevel 1 goto stopped
echo %date% %time% DONE step 3>> "%LOG%"

echo.
echo ===== Step 4 of 5: coherent stems, Slakh =====
echo %date% %time% START step 4 slakh coherent>> "%LOG%"
python data_tools\prepare_coherent.py slakh2100 urmp
if errorlevel 1 goto stopped
echo %date% %time% DONE step 4>> "%LOG%"

echo.
echo ===== Step 5 of 5: training =====
echo %date% %time% START step 5 training>> "%LOG%"
python training\train_separator.py
echo %date% %time% END step 5 training (finished or paused)>> "%LOG%"
exit /b 0

:stopped
echo %date% %time% STOPPED before finishing>> "%LOG%"
echo.
echo Data preparation stopped before finishing. Training was NOT started.
echo Run this file again to continue.
exit /b 1
