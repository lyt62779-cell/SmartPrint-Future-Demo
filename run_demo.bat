@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Install Python/Conda and follow README.md.
  pause
  exit /b 1
)
if not defined CNN_CLASSIFIER_PYTHON set "CNN_CLASSIFIER_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%CNN_CLASSIFIER_PYTHON%" set "CNN_CLASSIFIER_PYTHON=python"
python "%~dp0ocr_demo.py"
if errorlevel 1 pause
endlocal
