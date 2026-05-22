@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    echo Tao virtual environment...
    python -m venv .venv

    call .venv\Scripts\activate.bat

    echo Cai dat dependencies...
    pip install -q -r backend\requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo ============================================================
echo  Server: http://localhost:8000
echo  Mo trinh duyet sau khi model load xong (~30 giay).
echo ============================================================
echo.

start "" http://localhost:8000/?v=maxrows5000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 900
