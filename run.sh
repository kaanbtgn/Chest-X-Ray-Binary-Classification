#!/usr/bin/env bash
set -e

# — project folders —
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# — virtual-env —
if [[ ! -d "$PROJECT_DIR/venv" ]]; then
  echo "[+] Creating virtual environment (venv)…"
  python3 -m venv "$PROJECT_DIR/venv"
fi
source "$PROJECT_DIR/venv/bin/activate"

# — deps —
echo "[+] Installing Python packages (only first run takes time)…"
python -m pip install --upgrade pip wheel setuptools
pip install -r "$PROJECT_DIR/requirements.txt"

# — start FastAPI server —
echo "[+] Starting FastAPI server…"
python -m uvicorn app:app --host 0.0.0.0 --port 8501 --reload