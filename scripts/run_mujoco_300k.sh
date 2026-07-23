#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export LOGDIR="${LOGDIR:-logs_blcr_sac_locomotion_300k}"
export METHODS="${METHODS:-sac per_sac relo_sac blcr_sac}"
export ENVS="${ENVS:-HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5}"
export SEEDS="${SEEDS:-0 1 2}"
export TOTAL_STEPS="${TOTAL_STEPS:-300000}"
export LEARNING_STARTS="${LEARNING_STARTS:-5000}"
export EVAL_EVERY="${EVAL_EVERY:-10000}"
export EVAL_EPISODES="${EVAL_EPISODES:-5}"
export MAX_PARALLEL="${MAX_PARALLEL:-2}"
export NUM_THREADS="${NUM_THREADS:-1}"
export DEVICE="${DEVICE:-auto}"

bash scripts/run_blcr_sac_queue.sh
