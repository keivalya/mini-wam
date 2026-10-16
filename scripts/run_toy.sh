#!/usr/bin/env bash
# The paper's study on the 2D point-mass world: scripts/run.sh with the toy
# env, writing to results/seed*/ and outputs/wam/ as the study always has.
# Every setting of scripts/run.sh can still be changed, e.g.
#   RANDOM_ACTION_PROB=0.0 bash scripts/run_toy.sh
set -euo pipefail
ENV_NAME=toy OUT="${OUT:-results}" VIS_DIR="${VIS_DIR:-outputs/wam}" exec bash "$(dirname "$0")/run.sh"
