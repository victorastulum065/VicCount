#!/usr/bin/env bash
# VicCount — instalare (macOS / Linux). Rulează o singură dată, cu internet.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
  pip install torch torchvision
fi
pip install -r requirements.txt
python download_weights.py
echo
echo "Gata. Pornește cu: ./start.sh"
