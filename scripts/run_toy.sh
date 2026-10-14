#!/usr/bin/env bash
# End-to-end mini-WAM experiment on the toy world, repeated over several seeds:
#   for each seed: collect -> train WAM -> train VLA baseline -> evaluate -> closed-loop test
#   then: visualize seed 0, and summarize mean ± std over seeds
#
# Usage: bash scripts/run_toy.sh            (from the repo root)
# Outputs land in results/seed*/, outputs/ and videos/ inside the repo.
set -euo pipefail

# ======================= experiment configuration =======================
# Each experiment in the study changes one of these lines (and nothing else).
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-15}"
EPISODES="${EPISODES:-400}"              # demonstrations per seed (90% train / 10% held out)
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-0.3}"   # exploration bursts in the data
HORIZON="${HORIZON:-5}"                  # imagination steps per training window
STATE_WEIGHT="${STATE_WEIGHT:-0.5}"
IMAGE_WEIGHT="${IMAGE_WEIGHT:-1.0}"
LATENT_WEIGHT="${LATENT_WEIGHT:-1.0}"
STATE_INPUT="${STATE_INPUT:-1}"          # 0 = image + language only (no state_t input)
ACTION_HEAD="${ACTION_HEAD:-mlp}"
TEST_EPISODES="${TEST_EPISODES:-100}"
DEVICE="${DEVICE:-auto}"
# ========================================================================

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

echo "[run] host=$(hostname) python=$($PYV --version) torch=$($PYV -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")')"
echo "[run] cpus: $(nproc) slurm_job=${SLURM_JOB_ID:-none}"
echo "[run] config: SEEDS='$SEEDS' EPOCHS=$EPOCHS EPISODES=$EPISODES RANDOM_ACTION_PROB=$RANDOM_ACTION_PROB HORIZON=$HORIZON STATE_WEIGHT=$STATE_WEIGHT IMAGE_WEIGHT=$IMAGE_WEIGHT LATENT_WEIGHT=$LATENT_WEIGHT STATE_INPUT=$STATE_INPUT ACTION_HEAD=$ACTION_HEAD TEST_EPISODES=$TEST_EPISODES"

STATE_FLAG=""
[ "$STATE_INPUT" = "0" ] && STATE_FLAG="--no_state_input"

rm -rf results
for SEED in $SEEDS; do
  R="results/seed$SEED"
  mkdir -p "$R"
  echo "[run] ===== seed $SEED ====="

  # 1. data (a different dataset per seed)
  $PYV -m scripts.collect_data --env toy --episodes "$EPISODES" --max-steps 80 \
    --seed $((42 + 1000 * SEED)) --random-action-prob "$RANDOM_ACTION_PROB" \
    --output-path "$R/toy.npz" | tail -3

  # 2. WAM and the mini-VLA baseline (same data, same action head, same seed)
  $PYV -m scripts.train --dataset-path "$R/toy.npz" --model_type wam --action_head "$ACTION_HEAD" \
    --predict_state --predict_image $STATE_FLAG --horizon "$HORIZON" \
    --state-weight "$STATE_WEIGHT" --image-weight "$IMAGE_WEIGHT" --latent-weight "$LATENT_WEIGHT" \
    --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" --save-path "$R/wam.pt" | grep -E "Episodes:|Epoch $EPOCHS/| val:" | tail -3

  $PYV -m scripts.train --dataset-path "$R/toy.npz" --model_type vla --action_head "$ACTION_HEAD" \
    $STATE_FLAG --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" --save-path "$R/vla.pt" | grep -E "Epoch $EPOCHS/| val:" | tail -2

  # 3. world-model evaluation on held-out episodes (+ counterfactuals)
  $PYV -m scripts.evaluate --checkpoint "$R/wam.pt" --max-horizon 15 --device "$DEVICE" --output "$R/wam_eval.json"

  # 4. closed loop on unseen env seeds
  $PYV -m scripts.test --checkpoint "$R/wam.pt" --episodes "$TEST_EPISODES" --seed $((10000 + 1000 * SEED)) \
    --device cpu --json-out "$R/wam_test.json" | tail -1
  $PYV -m scripts.test --checkpoint "$R/vla.pt" --episodes "$TEST_EPISODES" --seed $((10000 + 1000 * SEED)) \
    --device cpu --json-out "$R/vla_test.json" | tail -1
done

# ---- 5. look inside the world model (first seed)
FIRST_SEED="${SEEDS%% *}"
$PYV -m scripts.visualize_rollout --checkpoint "results/seed$FIRST_SEED/wam.pt" --out-dir outputs/wam --video --device "$DEVICE"

# ---- 6. mean ± std over seeds
$PYV -m scripts.summarize_runs --root results

echo "[run] done"
