#!/usr/bin/env bash
set -euo pipefail

# Neurocomputing 2025-style MinAtar queue.
#
# Protocol defaults:
#   games: Breakout, Asterix, Seaquest, Freeway, SpaceInvaders
#   frames: 2M
#   seeds: 5
#   methods: DQN, PER, ReLo, ReLo-F, BLCR

GAMES="${GAMES:-breakout asterix seaquest freeway space_invaders}"
SEEDS="${SEEDS:-0 1 2 3 4}"
METHODS="${METHODS:-dqn per relo relo_floor blcr}"
NUM_FRAMES="${NUM_FRAMES:-2000000}"
LOGDIR="${LOGDIR:-logs_minatar_neurocomputing_2m}"
CONDA_ENV="${CONDA_ENV:-BLCR}"
PYTHON_BIN="${PYTHON_BIN:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
if [[ -z "${SLOT_SLEEP+x}" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    SLOT_SLEEP="1"
  else
    SLOT_SLEEP="10"
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

config_for_method() {
  local method="$1"
  case "$method" in
    dqn) echo "configs/minatar/dqn.yaml" ;;
    per) echo "configs/minatar/per.yaml" ;;
    relo) echo "configs/minatar/relo.yaml" ;;
    relo_floor) echo "configs/minatar/relo_floor.yaml" ;;
    relo_floor_adaptive) echo "configs/minatar/relo_floor_adaptive.yaml" ;;
    relo_floor_decay) echo "configs/minatar/relo_floor_decay.yaml" ;;
    blcr) echo "configs/minatar/blcr.yaml" ;;
    *)
      echo "ERROR: unsupported method: ${method}" >&2
      return 1
      ;;
  esac
}

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
    echo "ERROR: PYTHON_BIN is set but not executable: ${PYTHON_BIN}" >&2
    exit 127
  fi
else
  CONDA_BIN="$(resolve_conda || true)"
  if [[ -z "$CONDA_BIN" ]]; then
    echo "ERROR: neither PYTHON_BIN nor conda is available." >&2
    echo "Activate BLCR manually or set PYTHON_BIN=/path/to/python." >&2
    exit 127
  fi
fi

mkdir -p "$LOGDIR/_launcher"
RUN_LIST="$LOGDIR/_launcher/minatar_neurocomputing_run_list.tsv"
STATUS_DIR="$LOGDIR/_launcher/status"
mkdir -p "$STATUS_DIR"

printf "seed\tgame\tmethod\tconfig\tframes\n" > "$RUN_LIST"
for seed in $SEEDS; do
  for game in $GAMES; do
    for method in $METHODS; do
      config="$(config_for_method "$method")"
      printf "%s\t%s\t%s\t%s\t%s\n" "$seed" "$game" "$method" "$config" "$NUM_FRAMES" >> "$RUN_LIST"
    done
  done
done

run_one() {
  local seed="$1"
  local game="$2"
  local method="$3"
  local config="$4"
  local run_logdir="${LOGDIR}/${method}"
  local console_dir="${run_logdir}/console"
  local console_log="${console_dir}/${game}_seed${seed}.log"
  local result_path="${run_logdir}/${game}/${method}/seed${seed}/result.pt"
  local status_file="${STATUS_DIR}/${method}_${game}_seed${seed}.status"

  mkdir -p "$console_dir"

  if [[ "$SKIP_EXISTING" == "1" && -f "$result_path" ]]; then
    echo "SKIPPED" > "$status_file"
    echo "Skipping existing run: ${method}, ${game}, seed ${seed}" | tee "$console_log"
    return 0
  fi

  local -a cmd=()
  if [[ -n "$PYTHON_BIN" ]]; then
    cmd=("$PYTHON_BIN" -m fluid.train_minatar)
  else
    cmd=("$CONDA_BIN" run -n "$CONDA_ENV" python -m fluid.train_minatar)
  fi

  cmd+=(
    --config "$config"
    --game "$game"
    --seed "$seed"
    --algo "$method"
    --override "experiment.logdir=${run_logdir}"
    --override "training.num_frames=${NUM_FRAMES}"
  )

  echo "============================================================" | tee "$console_log"
  echo "Seed: ${seed}" | tee -a "$console_log"
  echo "Game: ${game}" | tee -a "$console_log"
  echo "Method: ${method}" | tee -a "$console_log"
  echo "Frames: ${NUM_FRAMES}" | tee -a "$console_log"
  echo "Logdir: ${run_logdir}" | tee -a "$console_log"
  echo "Command: ${cmd[*]}" | tee -a "$console_log"
  echo "Started: $(date -Is)" | tee -a "$console_log"
  echo "============================================================" | tee -a "$console_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN" > "$status_file"
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$console_log"
  local status=${PIPESTATUS[0]}
  set -e

  echo "Finished: $(date -Is)  ExitCode: ${status}" | tee -a "$console_log"
  echo "$status" > "$status_file"
  return "$status"
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    sleep "$SLOT_SLEEP"
  done
}

echo "Project root: ${PROJECT_ROOT}"
if [[ -n "$PYTHON_BIN" ]]; then
  echo "Python: ${PYTHON_BIN}"
else
  echo "Conda: ${CONDA_BIN}"
  echo "Conda env: ${CONDA_ENV}"
fi
echo "Methods: ${METHODS}"
echo "Games: ${GAMES}"
echo "Seeds: ${SEEDS}"
echo "Frames: ${NUM_FRAMES}"
echo "Log root: ${LOGDIR}"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Run list: ${RUN_LIST}"

while IFS=$'\t' read -r seed game method config frames; do
  [[ "$seed" == "seed" ]] && continue
  wait_for_slot
  (
    run_one "$seed" "$game" "$method" "$config"
  ) &
done < "$RUN_LIST"

set +e
wait
wait_status=$?
set -e

failures=0
for status_file in "$STATUS_DIR"/*.status; do
  [[ -e "$status_file" ]] || continue
  status="$(cat "$status_file")"
  if [[ "$status" != "0" && "$status" != "SKIPPED" && "$status" != "DRY_RUN" ]]; then
    failures=$((failures + 1))
  fi
done
if [[ "$wait_status" -ne 0 ]]; then
  failures=$((failures + 1))
fi

if [[ "$failures" -gt 0 ]]; then
  echo "MinAtar queue finished with ${failures} failed run(s)."
  if [[ "$STOP_ON_FAIL" == "1" ]]; then
    exit 1
  fi
fi

echo "MinAtar queue finished successfully."
echo "View results with: tensorboard --logdir ${LOGDIR}"
