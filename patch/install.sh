transformers_path=$(python -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo $transformers_path

cp patch/modeling_qwen2.py $transformers_path/models/qwen2
cp patch/modeling_qwen2_5_vl.py $transformers_path/models/qwen2_5_vl
cp patch/modeling_qwen3_vl.py $transformers_path/models/qwen3_vl
cp patch/modeling_llama.py $transformers_path/models/llama
cp patch/modeling_t5.py $transformers_path/models/t5
#
# IMPORTANT:
# Do NOT overwrite `transformers/trainer.py` here.
# Some environments ship a customized/older `transformers` build where
# `transformers.modeling_utils` may not provide symbols expected by newer trainer implementations
# (e.g. `load_sharded_checkpoint`). Overwriting trainer.py can break imports globally.
