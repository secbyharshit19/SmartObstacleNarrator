@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
.venv\Scripts\python.exe -m smart_obstacle_narrator.app --demo
pause
exit /b
:missing
echo Run SETUP_WINDOWS.bat first.
pause
