#!/usr/bin/env bash
set -e  # first error → exit

# — project folders —
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$PROJECT_DIR/chest_xray"
RUNS_DIR="$PROJECT_DIR/runs"

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

# — sanity checks —
[[ -d "$DATA_DIR" ]] || { echo "❌  Dataset not found: $DATA_DIR"; exit 1; }

# — training —
echo "[+] Starting training…"
python "$PROJECT_DIR/train_eval_tf.py" train \
  --data-dir "$DATA_DIR" \
  --epochs 25 \
  --batch-size 32 \
  --backbone densenet121

# — organise outputs —
mkdir -p "$RUNS_DIR"
mv checkpoints "$RUNS_DIR"/  2>/dev/null || true
mv logs "$RUNS_DIR"/  2>/dev/null || true

echo "
✅  Training finished.  See results inside:  $RUNS_DIR
To leave the virtual-env:  deactivate
"