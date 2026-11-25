@echo off
rem Raga Spatial - start the app (backend + frontend) and open it in the browser.
rem Double-click this file. Close the two windows it opens to stop the app.
rem If training is running, the app uses the CPU for the 8-stem model (slower).
cd /d "%~dp0"
start "Raga Spatial backend" cmd /k "call %~dp0env.bat && cd /d %~dp0backend && python -m uvicorn main:app --port 8000"
start "Raga Spatial frontend" cmd /k "call %~dp0env.bat && cd /d %~dp0frontend && npm run dev"
timeout /t 8 /nobreak >nul
start "" http://localhost:3000
