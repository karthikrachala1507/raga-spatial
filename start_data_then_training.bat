@echo off
rem Raga Spatial - finish preparing the training data, THEN train.
rem Use this instead of start_training.bat when datasets are still being
rem converted into clips: the two jobs slow each other down badly if they run
rem at the same time (both read heavily from the same disk).
rem
rem Step 1: data_tools\prepare_datasets.py  (skips everything already done)
rem Step 2: training\train_separator.py     (resumes from the last finished epoch)
rem
rem Pause at any time with Ctrl+C (answer Y). Run this file again to continue:
rem both steps pick up where they stopped.
call "C:\ProgramData\miniconda3\condabin\conda.bat" activate raga-spatial
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
cd /d "%~dp0backend"
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo.
echo ===== Step 1 of 2: preparing training clips =====
python data_tools\prepare_datasets.py
if errorlevel 1 (
    echo.
    echo Data preparation stopped before finishing. Training was NOT started.
    echo Run this file again to continue.
    exit /b 1
)

echo.
echo ===== Step 2 of 2: training =====
python training\train_separator.py
