#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-BLCR}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOGDIR="${LOGDIR:-logs_blcr_sac}"
METHODS="${METHODS:-sac per_sac relo_sac blcr_sac}"
ENVS="${ENVS:-HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5}"
SEEDS="${SEEDS:-0 1 2}"
TOTAL_STEPS="${TOTAL_STEPS:-300000}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
DEVICE="${DEVICE:-auto}"
NUM_THREADS="${NUM_THREADS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SLOT_SLEEP="${SLOT_SLEEP:-10}"

export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"

resolve_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  local candidate
  for candidate in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

CONDA_BIN=""
if [[ -n "$PYTHON_BIN" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 127
  fi
else
  CONDA_BIN="$(resolve_conda || true)"
  if [[ -z "$CONDA_BIN" ]]; then
    echo "ERROR: conda is unavailable; set PYTHON_BIN to an environment with BLCR installed." >&2
    exit 127
  fi
fi

mkdir -p "$LOGDIR/_launcher"
RUN_LIST="$LOGDIR/_launcher/blcr_sac_run_list.tsv"
: > "$RUN_LIST"

safe_name() {
  echo "$1" | tr '/:' '__'
}

for env in $ENVS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      env_safe="$(safe_name "$env")"
      run_dir="$LOGDIR/$method/$env_safe/seed$seed"
      printf "%s\t%s\t%s\t%s\n" "$env" "$method" "$seed" "$run_dir" >> "$RUN_LIST"
    done
  done
done

running_jobs() {
  jobs -rp | wc -l
}

launch_one() {
  local env="$1"
  local method="$2"
  local seed="$3"
  local run_dir="$4"
  mkdir -p "$run_dir"

  if [[ "$SKIP_EXISTING" == "1" && -f "$run_dir/result.pt" ]]; then
    echo "[skip] $method $env seed$seed already finished"
    return 0
  fi

  local -a cmd=()
  if [[ -n "$PYTHON_BIN" ]]; then
    cmd=("$PYTHON_BIN" -m fluid.train_blcr_sac)
  else
    cmd=("$CONDA_BIN" run -n "$CONDA_ENV" python -m fluid.train_blcr_sac)
  fi
  cmd+=(
    --env "$env"
    --method "$method"
    --seed "$seed"
    --total-steps "$TOTAL_STEPS"
    --learning-starts "$LEARNING_STARTS"
    --eval-every "$EVAL_EVERY"
    --eval-episodes "$EVAL_EPISODES"
    --device "$DEVICE"
    --num-threads "$NUM_THREADS"
    --logdir "$LOGDIR"
  )

  echo "[start] $method $env seed$seed"
  (
    "${cmd[@]}" > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
    echo "done" > "$run_dir/status.txt"
  ) &
}

while IFS=$'\t' read -r env method seed run_dir; do
  while [[ "$(running_jobs)" -ge "$MAX_PARALLEL" ]]; do
    sleep "$SLOT_SLEEP"
  done
  launch_one "$env" "$method" "$seed" "$run_dir"
done < "$RUN_LIST"

wait
echo "[all done] wrote runs to $LOGDIR"
