#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"

transformers_path=$("${python_bin}" -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo $transformers_path

cp "${script_dir}/modeling_qwen2.py" "$transformers_path/models/qwen2"
cp "${script_dir}/modeling_qwen2_5_vl.py" "$transformers_path/models/qwen2_5_vl"
cp "${script_dir}/modeling_qwen3_vl.py" "$transformers_path/models/qwen3_vl"
cp "${script_dir}/modeling_llama.py" "$transformers_path/models/llama"
cp "${script_dir}/modeling_t5.py" "$transformers_path/models/t5"
#
# IMPORTANT:
# Do NOT overwrite `transformers/trainer.py` here.
# Some environments ship a customized/older `transformers` build where
# `transformers.modeling_utils` may not provide symbols expected by newer trainer implementations
# (e.g. `load_sharded_checkpoint`). Overwriting trainer.py can break imports globally.
