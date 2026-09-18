@echo off
chcp 65001 >nul
cd /d "%~dp0"
title ASTK Web

if not exist ".venv\Scripts\python.exe" (
    echo [!] First run: creating .venv and installing dependencies. This takes a few minutes...
    python -m venv .venv
    if errorlevel 1 goto err
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto err
)

echo.
echo [OK] Starting ASTK Web ... a browser tab will open at http://localhost:8501
echo      Press Ctrl+C in this window to stop.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
echo.
echo Server stopped.
pause
exit /b 0

:err
echo.
echo [X] Setup failed. Make sure Python 3.10+ is installed and on PATH
echo     (https://www.python.org/downloads/ -> tick "Add python.exe to PATH").
pause
exit /b 1