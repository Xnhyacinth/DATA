#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"

py_package_path=$("${python_bin}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo $py_package_path
"${python_bin}" "${script_dir}/remove_patch.py" --package_path "$py_package_path/transformers/models/llama" --patch_files modeling_llama.py
"${python_bin}" "${script_dir}/remove_patch.py" --package_path "$py_package_path/transformers/models/qwen2_5_vl" --patch_files modeling_qwen2_5_vl.py
"${python_bin}" "${script_dir}/remove_patch.py" --package_path "$py_package_path/transformers/models/qwen3_vl" --patch_files modeling_qwen3_vl.py
"${python_bin}" "${script_dir}/remove_patch.py" --package_path "$py_package_path/transformers/models/t5" --patch_files modeling_t5.py
# We no longer patch `transformers/trainer.py` in install.sh.

trainer_path="$py_package_path/transformers/trainer.py"
"${python_bin}" - "$trainer_path" <<'PY'
import pathlib
import sys

trainer_path = pathlib.Path(sys.argv[1])
if not trainer_path.exists():
    raise SystemExit(0)

content = trainer_path.read_text(encoding="utf-8")
target = "from .modeling_utils import PreTrainedModel, load_sharded_checkpoint"
replacement = """try:
    from .modeling_utils import PreTrainedModel, load_sharded_checkpoint
except ImportError:
    from .modeling_utils import PreTrainedModel

    def load_sharded_checkpoint(*args, **kwargs):
        raise ImportError(
            \"transformers.modeling_utils.load_sharded_checkpoint is not available in this environment.\"
        )
"""

if target in content and replacement not in content:
    trainer_path.write_text(content.replace(target, replacement), encoding="utf-8")
    print(f\"Applied compatibility hotfix to: {trainer_path}\")
PY
