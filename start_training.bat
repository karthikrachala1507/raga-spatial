@echo off
rem Raga Spatial - start or resume training of the 8-stem separator.
rem Double-click this file, or run it from any terminal.
rem Pause: press Ctrl+C (answer Y). Resume: run this file again.
rem It continues from the last finished epoch (data\checkpoints\separator_v3_latest.pt).
call "%~dp0env.bat"
if errorlevel 1 (
    echo ERROR: could not set up the project environment, see env.bat.
    exit /b 1
)
cd /d "%~dp0backend"
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python training\train_separator.py
