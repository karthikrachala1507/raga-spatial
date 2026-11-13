@echo off
rem Raga Spatial - start or resume training of the 8-stem separator.
rem Double-click this file, or run it from any terminal.
rem Pause: press Ctrl+C (answer Y). Resume: run this file again.
rem It continues from the last finished epoch (data\checkpoints\separator_v3_latest.pt).
call "C:\ProgramData\miniconda3\condabin\conda.bat" activate raga-spatial
if errorlevel 1 (
    echo ERROR: could not activate the raga-spatial conda environment.
    exit /b 1
)
cd /d "%~dp0backend"
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python training\train_separator.py
