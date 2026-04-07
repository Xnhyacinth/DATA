py_package_path=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo $py_package_path
python remove_patch.py --package_path $py_package_path/transformers/models/llama --patch_files modeling_llama.py
python remove_patch.py --package_path $py_package_path/transformers/models/qwen2_5_vl --patch_files modeling_qwen2_5_vl.py
python remove_patch.py --package_path $py_package_path/transformers/models/qwen3_vl --patch_files modeling_qwen3_vl.py
python remove_patch.py --package_path $py_package_path/transformers/models/t5 --patch_files modeling_t5.py
python remove_patch.py --package_path $py_package_path/transformers --patch_files trainer.py
