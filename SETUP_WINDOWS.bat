@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto failure
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failure
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failure
echo Setup complete. Run RUN_BASELINE.bat to start the webcam.
pause
exit /b 0
:failure
echo Setup failed. Read the error above and docs\TROUBLESHOOTING.md.
pause
exit /b 1
