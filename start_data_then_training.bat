@echo off
rem Raga Spatial - finish preparing the training data, THEN train.
rem Use this instead of start_training.bat when data is still being prepared:
rem the jobs slow each other down badly if they run at the same time (they all
rem read heavily from the same disk).
rem
rem Step 1: data_tools\prepare_datasets.py  single clips for random mixes
rem Step 2: data_tools\prepare_coherent.py  time-aligned stems for coherent mixes
rem Step 3: training\train_separator.py     resumes from the last finished epoch
rem
rem Every step skips what is already done. Pause at any time with Ctrl+C
rem (answer Y) and run this file again to continue.
call "C:\ProgramData\miniconda3\condabin\conda.bat" activate raga-spatial
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
cd /d "%~dp0backend"
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo.
echo ===== Step 1 of 3: single clips =====
python data_tools\prepare_datasets.py
if errorlevel 1 goto stopped

echo.
echo ===== Step 2 of 3: time-aligned stems for coherent mixes =====
python data_tools\prepare_coherent.py
if errorlevel 1 goto stopped

echo.
echo ===== Step 3 of 3: training =====
python training\train_separator.py
exit /b 0

:stopped
echo.
echo Data preparation stopped before finishing. Training was NOT started.
echo Run this file again to continue.
exit /b 1
