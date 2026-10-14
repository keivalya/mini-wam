#!/usr/bin/env bash
# End-to-end mini-WAM experiment on the toy world:
#   collect -> train WAM -> train VLA baseline -> evaluate -> visualize -> closed-loop test
#
# Usage: bash scripts/run_toy.sh            (from the repo root)
# Outputs land in data/, checkpoints/, outputs/ and videos/ inside the repo.
set -euo pipefail

EPOCHS="${EPOCHS:-15}"
EPISODES="${EPISODES:-400}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"

# ---- environment: one venv per package list, reused across runs
#      (set MINI_WAM_PY=/path/to/python to use an existing environment instead)
PKGS='torch numpy opencv-python-headless imageio imageio[ffmpeg] matplotlib'
PY="${PYTHON:-python3}"

if command -v module >/dev/null 2>&1 && [ -n "${MINI_WAM_MODULES:-}" ]; then
  module load $MINI_WAM_MODULES
fi

if [ -n "${MINI_WAM_PY:-}" ]; then
  PYV="$MINI_WAM_PY"
else
  PKG_HASH="$(echo "$PKGS $($PY --version 2>&1)" | cksum | cut -d' ' -f1)"
  VENV="$HOME/.cache/mini-wam/venv-$PKG_HASH"
  if [ ! -x "$VENV/bin/python" ]; then
    echo "[run] creating venv $VENV with $($PY --version 2>&1)"
    mkdir -p "$(dirname "$VENV")"
    if command -v uv >/dev/null 2>&1; then
      uv venv "$VENV" --python "$PY"
      VIRTUAL_ENV="$VENV" uv pip install $PKGS
    else
      "$PY" -m venv "$VENV"
      "$VENV/bin/pip" install --upgrade pip
      "$VENV/bin/pip" install $PKGS
    fi
  fi
  PYV="$VENV/bin/python"
fi
export PYTHONUNBUFFERED=1

echo "[run] host=$(hostname) python=$($PYV --version) torch=$($PYV -c 'import torch; print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")')"
echo "[run] git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[run] config: EPOCHS=$EPOCHS EPISODES=$EPISODES SEED=$SEED DEVICE=$DEVICE"

# ---- 1. data
$PYV -m scripts.collect_data --env toy --episodes "$EPISODES" --max-steps 80 \
  --seed 42 --output-path data/toy.npz | tail -12

# ---- 2. WAM and the mini-VLA baseline (same data, same action head, same seed)
$PYV -m scripts.train --dataset-path data/toy.npz --model_type wam --action_head mlp \
  --predict_state --predict_image --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" \
  --save-path checkpoints/wam.pt

$PYV -m scripts.train --dataset-path data/toy.npz --model_type vla --action_head mlp \
  --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" --save-path checkpoints/vla.pt

# ---- 3. world-model evaluation on held-out episodes
$PYV -m scripts.evaluate --checkpoint checkpoints/wam.pt --max-horizon 15

# ---- 4. look inside the world model
$PYV -m scripts.visualize_rollout --checkpoint checkpoints/wam.pt --out-dir outputs/wam --video

# ---- 5. closed loop on unseen seeds
echo "[run] closed-loop WAM"
$PYV -m scripts.test --checkpoint checkpoints/wam.pt --episodes 50 --device cpu --save-video --video-dir videos/wam | tail -1
echo "[run] closed-loop VLA"
$PYV -m scripts.test --checkpoint checkpoints/vla.pt --episodes 50 --device cpu | tail -1

echo "[run] done"
