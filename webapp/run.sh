#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Tạo virtual environment..."
    python3 -m venv .venv
    INSTALL_DEPS=1
else
    INSTALL_DEPS=0
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ "$INSTALL_DEPS" = "1" ]; then
    echo "Cài đặt dependencies..."
    pip install -q -r backend/requirements.txt
fi

echo
echo "============================================================"
echo " Server: http://localhost:8000"
echo " Mở trình duyệt sau khi model load xong (~30 giây)."
echo "============================================================"
echo

(sleep 3 && (xdg-open "http://localhost:8000/?v=maxrows5000" 2>/dev/null || open "http://localhost:8000/?v=maxrows5000" 2>/dev/null)) &
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 900
