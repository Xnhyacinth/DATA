#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"

transformers_path=$("${python_bin}" -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo "$transformers_path"

declare -a patch_specs=(
  "qwen2:modeling_qwen2.py"
  "qwen2_5_vl:modeling_qwen2_5_vl.py"
  "qwen3_vl:modeling_qwen3_vl.py"
  "llama:modeling_llama.py"
  "t5:modeling_t5.py"
)

for spec in "${patch_specs[@]}"; do
  package_name="${spec%%:*}"
  patch_file="${spec#*:}"
  "${python_bin}" "${script_dir}/run_patch.py" \
    --package_path "${transformers_path}/models/${package_name}" \
    --patch_files "${patch_file}"
done

#
# IMPORTANT:
# Do NOT overwrite `transformers/trainer.py` here.
# Some environments ship a customized/older `transformers` build where
# `transformers.modeling_utils` may not provide symbols expected by newer trainer implementations
# (e.g. `load_sharded_checkpoint`). Overwriting trainer.py can break imports globally.
