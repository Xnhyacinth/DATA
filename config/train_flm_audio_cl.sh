#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for CL training with:
#   CofeAI/FLM-Audio
#
# This script delegates to `config/train.sh` (LLaMA-Factory `--stage cl`).
#
# Usage examples:
#   bash config/train_flm_audio_cl.sh 2 0,1 order_1 lora 1e-4 8 42
#   bash config/train_flm_audio_cl.sh 2 2,3 order_4 data 1e-4 8 42
#
# Args (all optional, in order):
#   1: num_gpus (default: 2)
#   2: gpus     (default: 0,1)
#   3: order    (default: order_1)
#   4: tuning   (default: lora)      # lora | data | full
#   5: lr       (default: 1e-4)
#   6: lora_r   (default: 8)
#   7: seed     (default: 42)

num_gpus="${1:-2}"
gpus="${2:-0,1}"
order="${3:-order_1}"
tuning="${4:-lora}"
lr="${5:-1e-4}"
lora_r="${6:-8}"
seed="${7:-42}"

# Advanced defaults (override via env vars if needed).
epoch="${EPOCH:-1}"
bs="${BS:-1}"
template="${TEMPLATE:-flm_audio}"
lr_scheduler_type="${LR_SCHEDULER_TYPE:-constant}"
deepspeed="${DEEPSPEED:--1}"
val_size="${VAL_SIZE:-1e-10}"
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

# Logging / eval hook defaults.
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export REPORT_TO="${REPORT_TO:-none}"
export TEST_AFTER_EACH="${TEST_AFTER_EACH:-1}"
export TEST_AFTER_EACH_SCOPE="${TEST_AFTER_EACH_SCOPE:-seen}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/train.sh" \
  "${num_gpus}" \
  "${gpus}" \
  "flm-audio" \
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
