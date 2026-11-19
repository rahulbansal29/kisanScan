@echo off
cd /d "%~dp0"
echo Starting KisanScan... wait for "Model ready", then the browser opens.
start "" cmd /c "timeout /t 12 >nul && start http://localhost:5000"
.venv\Scripts\python app.py
pause
