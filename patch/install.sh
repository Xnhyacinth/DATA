pip install transformers==4.45.2
transformers_path=$(python -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo $transformers_path

cp patch/modeling_qwen2.py $transformers_path/models/qwen2
cp patch/modeling_llama.py $transformers_path/models/llama
cp patch/modeling_t5.py $transformers_path/models/t5
cp patch/trainer.py $transformers_path
