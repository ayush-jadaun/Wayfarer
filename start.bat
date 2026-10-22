@echo off
rem Wayfarer: double-click to start the job server + web app and open the upload studio.
cd /d "%~dp0"

if not exist ".venv-ns\Scripts\python.exe" (
  echo Training environment missing. Follow docs\SETUP.md first.
  pause
  exit /b 1
)
rem COLMAP 4.2+ first: older installs on PATH lack the global mapper
if exist "E:\tools\colmap4\COLMAP.bat" set "PATH=E:\tools\colmap4;%PATH%"
where colmap >nul 2>nul || echo WARNING: COLMAP not on PATH - see docs\SETUP.md

if not exist "web\node_modules" (
  echo Installing web dependencies...
  pushd web && call npm install && popd
)

start "Wayfarer job server" cmd /k ".venv-ns\Scripts\python.exe pipeline\server.py"
start "Wayfarer web" cmd /k "cd web && npm run dev -- --host 127.0.0.1 --port 5173"
timeout /t 4 >nul
start "" "http://127.0.0.1:5173/upload.html"
