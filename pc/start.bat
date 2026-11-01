@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call .venv\Scripts\activate.bat
python server.py %*
pause
