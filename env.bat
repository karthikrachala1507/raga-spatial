@echo off
rem Raga Spatial - shared setup. Python, Node, caches and temp files all come from
rem this project folder on D:; the Miniconda and Node installed on C: are not used.
rem Every start_*.bat calls this first. For a terminal:  call D:\raga-spatial\env.bat
set "RAGA_ROOT=%~dp0"
set "RAGA_ENV=%~dp0env"
set "RAGA_CACHE=%~dp0data\cache"
if not exist "%RAGA_ENV%\python.exe" (
    echo ERROR: Python environment not found at %RAGA_ENV%
    exit /b 1
)
if not exist "%RAGA_CACHE%\tmp" mkdir "%RAGA_CACHE%\tmp"
set "TEMP=%RAGA_CACHE%\tmp"
set "TMP=%RAGA_CACHE%\tmp"
set "HF_HOME=%RAGA_CACHE%\huggingface"
set "TORCH_HOME=%RAGA_CACHE%\torch"
set "PIP_CACHE_DIR=%RAGA_CACHE%\pip"
set "npm_config_cache=%RAGA_CACHE%\npm"
set "npm_config_prefix=%~dp0tools\npm-global"
set "NUMBA_CACHE_DIR=%~dp0data\numba_cache"
if not exist "%NUMBA_CACHE_DIR%" mkdir "%NUMBA_CACHE_DIR%"
rem ignore packages in the user folder on C:
set "PYTHONNOUSERSITE=1"

rem what "conda activate" would do, without needing conda
set "CONDA_PREFIX=%RAGA_ENV%"
rem First on PATH: the bundled ffmpeg (the environment's own ffmpeg.exe does not
rem start and shows an "Entry Point Not Found" box).
set "PATH=%RAGA_ENV%\Lib\site-packages\static_ffmpeg\bin\win32;%RAGA_ENV%;%RAGA_ENV%\Library\mingw-w64\bin;%RAGA_ENV%\Library\usr\bin;%RAGA_ENV%\Library\bin;%RAGA_ENV%\Scripts;%RAGA_ENV%\bin;%~dp0tools\node;%PATH%"
for %%f in ("%RAGA_ENV%\etc\conda\activate.d\*.bat") do call "%%f"
exit /b 0
