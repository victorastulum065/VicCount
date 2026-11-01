@echo off
REM VicCount - instalare Windows. Ruleaza o singura data, cu internet.
cd /d "%~dp0"
set PYTHONUTF8=1
py -3 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -U pip wheel
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
  pip install torch torchvision
)
pip install -r requirements.txt
python download_weights.py
echo.
echo Gata. Porneste cu start.bat
pause
