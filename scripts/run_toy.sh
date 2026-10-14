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

# CPU-only torch wheels unless this machine has an NVIDIA GPU (much smaller install)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

if [ -n "${MINI_WAM_PY:-}" ]; then
  PYV="$MINI_WAM_PY"
else
  if command -v uv >/dev/null 2>&1; then
    PY_SPEC="${UV_PYTHON_VERSION:-3.11}"      # uv fetches this Python if needed
  else
    PY_SPEC="$($PY --version 2>&1)"
  fi
  PKG_HASH="$(echo "$PKGS $PY_SPEC $TORCH_INDEX" | cksum | cut -d' ' -f1)"
  VENV="$HOME/.cache/mini-wam/venv-$PKG_HASH"
  # .ready is written only after a complete install, so an interrupted
  # install is rebuilt instead of silently reused
  if [ ! -f "$VENV/.ready" ]; then
    rm -rf "$VENV"
    echo "[run] creating venv $VENV (python $PY_SPEC, torch from $TORCH_INDEX)"
    mkdir -p "$(dirname "$VENV")"
    if command -v uv >/dev/null 2>&1; then
      uv venv "$VENV" --python "$PY_SPEC"
      VIRTUAL_ENV="$VENV" uv pip install torch --index-url "$TORCH_INDEX"
      VIRTUAL_ENV="$VENV" uv pip install $PKGS
    else
      "$PY" -m venv "$VENV"
      "$VENV/bin/pip" install --upgrade pip
      "$VENV/bin/pip" install torch --index-url "$TORCH_INDEX"
      "$VENV/bin/pip" install $PKGS
    fi
    touch "$VENV/.ready"
  fi
  PYV="$VENV/bin/python"
fi
export PYTHONUNBUFFERED=1

echo "[run] host=$(hostname) python=$($PYV --version) torch=$($PYV -c 'import torch; print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")')"
echo "[run] git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[run] cpus: $(nproc) slurm_job=${SLURM_JOB_ID:-none}"
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
