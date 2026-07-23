#!/usr/bin/env bash
set -euo pipefail

# ALE RAM comparison queue:
#   baseline: relo
#   proposed method: BLCR
#
# Default workload:
#   30 ALE games x 2 methods x 5 seeds x 2M frames = 300 runs
#
# This script uses RAM observations through Gymnasium/ALE. It does not run
# pixel-based Atari CNN experiments.

GAMES="${GAMES:-alien amidar assault asterix asteroids atlantis bank_heist battle_zone beam_rider bowling boxing breakout centipede chopper_command crazy_climber demon_attack enduro fishing_derby freeway frostbite gopher gravitar hero ice_hockey jamesbond kangaroo krull kung_fu_master montezuma_revenge ms_pacman}"
METHODS="${METHODS:-relo blcr}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_FRAMES="${NUM_FRAMES:-2000000}"
LOGDIR="${LOGDIR:-logs_ale_ram_30games}"
CONDA_ENV="${CONDA_ENV:-BLCR}"
PYTHON_BIN="${PYTHON_BIN:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
SLOT_SLEEP="${SLOT_SLEEP:-10}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

config_for_method() {
  local method="$1"
  case "$method" in
    relo) echo "configs/external/ale_ram_relo.yaml" ;;
    blcr) echo "configs/external/ale_ram_blcr.yaml" ;;
    *)
      echo "ERROR: unsupported method: ${method}" >&2
      return 1
      ;;
  esac
}

ale_env_id() {
  local game="$1"
  case "$game" in
    alien) echo "ALE/Alien-v5" ;;
    amidar) echo "ALE/Amidar-v5" ;;
    assault) echo "ALE/Assault-v5" ;;
    asterix) echo "ALE/Asterix-v5" ;;
    asteroids) echo "ALE/Asteroids-v5" ;;
    atlantis) echo "ALE/Atlantis-v5" ;;
    bank_heist) echo "ALE/BankHeist-v5" ;;
    battle_zone) echo "ALE/BattleZone-v5" ;;
    beam_rider) echo "ALE/BeamRider-v5" ;;
    bowling) echo "ALE/Bowling-v5" ;;
    boxing) echo "ALE/Boxing-v5" ;;
    breakout) echo "ALE/Breakout-v5" ;;
    centipede) echo "ALE/Centipede-v5" ;;
    chopper_command) echo "ALE/ChopperCommand-v5" ;;
    crazy_climber) echo "ALE/CrazyClimber-v5" ;;
    demon_attack) echo "ALE/DemonAttack-v5" ;;
    enduro) echo "ALE/Enduro-v5" ;;
    fishing_derby) echo "ALE/FishingDerby-v5" ;;
    freeway) echo "ALE/Freeway-v5" ;;
    frostbite) echo "ALE/Frostbite-v5" ;;
    gopher) echo "ALE/Gopher-v5" ;;
    gravitar) echo "ALE/Gravitar-v5" ;;
    hero) echo "ALE/Hero-v5" ;;
    ice_hockey) echo "ALE/IceHockey-v5" ;;
    jamesbond) echo "ALE/Jamesbond-v5" ;;
    kangaroo) echo "ALE/Kangaroo-v5" ;;
    krull) echo "ALE/Krull-v5" ;;
    kung_fu_master) echo "ALE/KungFuMaster-v5" ;;
    montezuma_revenge) echo "ALE/MontezumaRevenge-v5" ;;
    ms_pacman) echo "ALE/MsPacman-v5" ;;
    *)
      echo "ERROR: unsupported ALE game key: ${game}" >&2
      return 1
      ;;
  esac
}

task_name_for_game() {
  local game="$1"
  echo "ale_${game}_ram"
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
RUN_LIST="$LOGDIR/_launcher/ale_ram_30games_run_list.tsv"
STATUS_DIR="$LOGDIR/_launcher/status"
mkdir -p "$STATUS_DIR"

printf "seed\tgame\ttask\tmethod\tenv_id\tconfig\tframes\n" > "$RUN_LIST"
for seed in $SEEDS; do
  for game in $GAMES; do
    env_id="$(ale_env_id "$game")"
    task="$(task_name_for_game "$game")"
    for method in $METHODS; do
      config="$(config_for_method "$method")"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$seed" "$game" "$task" "$method" "$env_id" "$config" "$NUM_FRAMES" >> "$RUN_LIST"
    done
  done
done

run_one() {
  local seed="$1"
  local game="$2"
  local task="$3"
  local method="$4"
  local env_id="$5"
  local config="$6"
  local run_logdir="${LOGDIR}/${method}"
  local console_dir="${run_logdir}/console"
  local console_log="${console_dir}/${task}_seed${seed}.log"
  local result_path="${run_logdir}/${task}/${method}/seed${seed}/result.pt"
  local status_file="${STATUS_DIR}/${method}_${task}_seed${seed}.status"

  mkdir -p "$console_dir"

  if [[ "$SKIP_EXISTING" == "1" && -f "$result_path" ]]; then
    echo "SKIPPED" > "$status_file"
    echo "Skipping existing run: ${method}, ${task}, seed ${seed}" | tee "$console_log"
    return 0
  fi

  local -a cmd=()
  if [[ -n "$PYTHON_BIN" ]]; then
    cmd=("$PYTHON_BIN" -m fluid.train_external_discrete)
  else
    cmd=("$CONDA_BIN" run -n "$CONDA_ENV" python -m fluid.train_external_discrete)
  fi

  cmd+=(
    --config "$config"
    --seed "$seed"
    --algo "$method"
    --env-name "$task"
    --override "env.id=${env_id}"
    --override "experiment.logdir=${run_logdir}"
    --override "training.num_frames=${NUM_FRAMES}"
  )

  echo "============================================================" | tee "$console_log"
  echo "Seed: ${seed}" | tee -a "$console_log"
  echo "Game: ${game}" | tee -a "$console_log"
  echo "Task: ${task}" | tee -a "$console_log"
  echo "ALE env: ${env_id}" | tee -a "$console_log"
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

while IFS=$'\t' read -r seed game task method env_id config frames; do
  [[ "$seed" == "seed" ]] && continue
  wait_for_slot
  (
    run_one "$seed" "$game" "$task" "$method" "$env_id" "$config"
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
  echo "ALE RAM queue finished with ${failures} failed run(s)."
  if [[ "$STOP_ON_FAIL" == "1" ]]; then
    exit 1
  fi
fi

echo "ALE RAM queue finished successfully."
echo "View results with: tensorboard --logdir ${LOGDIR}"
