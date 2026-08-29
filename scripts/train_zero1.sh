#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=("$@")
NPROC_PER_NODE="${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-8}}"
NUM_MACHINES="${NNODES:-${PET_NNODES:-1}}"
MACHINE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
MAIN_PROCESS_IP="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
MAIN_PROCESS_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
GLOBAL_NUM_PROCESSES=$((NPROC_PER_NODE * NUM_MACHINES))

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[launch] NPROC_PER_NODE/PET_NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 2
fi
if ! [[ "${NUM_MACHINES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[launch] NNODES/PET_NNODES must be a positive integer, got: ${NUM_MACHINES}" >&2
  exit 2
fi
if ! [[ "${MACHINE_RANK}" =~ ^[0-9]+$ ]] || (( MACHINE_RANK >= NUM_MACHINES )); then
  echo "[launch] NODE_RANK/PET_NODE_RANK must be in [0, $((NUM_MACHINES - 1))], got: ${MACHINE_RANK}" >&2
  exit 2
fi

if (( NUM_MACHINES > 1 )); then
  if [[ -z "${MAIN_PROCESS_IP}" ]]; then
    echo "[launch] MASTER_ADDR or PET_MASTER_ADDR is required for multi-machine training." >&2
    exit 2
  fi
else
  MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
fi
if ! [[ "${MAIN_PROCESS_PORT}" =~ ^[0-9]+$ ]] || (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
  echo "[launch] MASTER_PORT/PET_MASTER_PORT must be in [1, 65535], got: ${MAIN_PROCESS_PORT}" >&2
  exit 2
fi

TASK_BASENAME="train"
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == task=* ]]; then
    TASK_BASENAME="${arg#task=}"
    TASK_BASENAME="${TASK_BASENAME%.yaml}"
  fi
done

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_processes=${GLOBAL_NUM_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${GLOBAL_NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --deepspeed_multinode_launcher standard \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
