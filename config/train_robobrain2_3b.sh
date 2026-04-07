#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for training on DATA (Long Sequence Benchmark) with:
#   BAAI/RoboBrain2.0-3B
#
# This script delegates to `config/train.sh` (LLaMA-Factory `--stage cl`).
#
# Usage examples:
#   bash config/train_robobrain2_3b.sh 2 0,1 order_4 lora 1e-4 8 8 42
#   bash config/train_robobrain2_3b.sh 1 0 order_4 lora 2e-4 2 8 42
#
# Args (all optional, in order):
#   1: num_gpus (default: 2)
#   2: gpus     (default: 0,1)
#   3: order    (default: order_4)   # order_4/5/6 are the 15-task CL orders in `config/train.sh`
#   4: tuning   (default: lora)      # lora | data | full
#   5: lr       (default: 1e-4)
#   6: bs       (default: 2)         # per-device train batch size
#   7: lora_r   (default: 8)
#   8: seed     (default: 42)
#
# You can still override advanced knobs via env vars below.

num_gpus="${1:-2}"
gpus="${2:-0,1}"
order="${3:-order_4}"
tuning="${4:-lora}"
lr="${5:-1e-4}"
bs="${6:-2}"
lora_r="${7:-8}"
seed="${8:-42}"

# Advanced defaults (override via env vars if needed).
epoch="${EPOCH:-1}"
template="${TEMPLATE:-fewshot}"
lr_scheduler_type="${LR_SCHEDULER_TYPE:-constant}"
deepspeed="${DEEPSPEED:-"-1"}"
val_size="${VAL_SIZE:-"1e-10"}"
filter="${FILTER:-0}"
mode="${MODE:-all}"
select="${SELECT:-0}"
data_rank1="${DATA_RANK1:-4}"
data_rank2="${DATA_RANK2:-16}"
restore="${RESTORE:-0}"
scale="${SCALE:-0}"
adaprompt="${ADAPROMPT:-0}"
reinit="${REINIT:-0}"
ortho_mu="${ORTHO_MU:-0}"
gap_layers="${GAP_LAYERS:-4}"
bakebone="${BAKEBONE:-0}"
nomlp="${NOMLP:-0}"
project="${PROJECT:-0}"
replay="${REPLAY:-0}"
max_samples="${MAX_SAMPLES:-1000000}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/train.sh" \
  "${num_gpus}" \
  "${gpus}" \
  "robobrain2-3b" \
  "${order}" \
  "${tuning}" \
  "${epoch}" \
  "${lr}" \
  "${bs}" \
  "${template}" \
  "${deepspeed}" \
  "${val_size}" \
  "${lora_r}" \
  "${lr_scheduler_type}" \
  "${seed}" \
  "${filter}" \
  "${mode}" \
  "${select}" \
  "${data_rank1}" \
  "${data_rank2}" \
  "${restore}" \
  "${scale}" \
  "${adaprompt}" \
  "${reinit}" \
  "${ortho_mu}" \
  "${gap_layers}" \
  "${bakebone}" \
  "${nomlp}" \
  "${project}" \
  "${replay}" \
  "${max_samples}"

