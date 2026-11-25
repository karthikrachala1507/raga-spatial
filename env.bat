@echo off
rem Raga Spatial - shared setup: the Python environment, caches and temp files,
rem all inside the project folder on D:. Every start_*.bat calls this first.
set "RAGA_ROOT=%~dp0"
set "RAGA_CACHE=%~dp0data\cache"
if not exist "%RAGA_CACHE%\tmp" mkdir "%RAGA_CACHE%\tmp"
set "TEMP=%RAGA_CACHE%\tmp"
set "TMP=%RAGA_CACHE%\tmp"
set "HF_HOME=%RAGA_CACHE%\huggingface"
set "TORCH_HOME=%RAGA_CACHE%\torch"
set "PIP_CACHE_DIR=%RAGA_CACHE%\pip"
set "npm_config_cache=%RAGA_CACHE%\npm"
set "NUMBA_CACHE_DIR=%~dp0data\numba_cache"
if not exist "%NUMBA_CACHE_DIR%" mkdir "%NUMBA_CACHE_DIR%"
call "C:\ProgramData\miniconda3\condabin\conda.bat" activate "%~dp0env"
rem The environment's own ffmpeg.exe does not start (it shows an "Entry Point Not
rem Found" box), so the working bundled one goes first on PATH.
set "PATH=%~dp0env\Lib\site-packages\static_ffmpeg\bin\win32;%PATH%"
