ini_thresholds=(0.7)
model="google-t5/t5-large" 
method="baseline" # migu
seeds=(42) #  1024 2048
seeds=(42) #  1024 2048

gpus=${1:-"1"}
model_name_or_path=${2:-"t5-large"}
lr=${3:-"1e-4"}
lr_type=${4:-"constant"}
tuning_method=${5:-"lora_tuning"}
method=${6:-"baseline"}

if [ "$model_name_or_path" = "tinyllama" ];then
    model=TinyLlama/TinyLlama_v1.1
fi
if [ "$model_name_or_path" = "llama2-7b" ];then
    model=meta-llama/Llama-2-7b-hf
fi
if [ "$model_name_or_path" = "t5-large" ];then
    model=google-t5/t5-large
fi
if [ "$model_name_or_path" = "qwen2.5-0.5b" ];then
    model=Qwen/Qwen2.5-0.5B
fi
if [ "$model_name_or_path" = "qwen2.5-1.5b" ];then
    model=Qwen/Qwen2.5-1.5B
fi
if [ "$model_name_or_path" = "qwen2.5-3b" ];then
    model=Qwen/Qwen2.5-3B
fi
if [ "$model_name_or_path" = "qwen2.5-7b" ];then
    model=Qwen/Qwen2.5-7B
fi
if [ "$model_name_or_path" = "llama3-8b" ];then
    model=meta-llama/Meta-Llama-3-8B
fi
if [ "$model_name_or_path" = "llama3.1-8b" ];then
    model=meta-llama/Llama-3.1-8B
fi
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for ini_threshold in "${ini_thresholds[@]}"; do
  for seed in "${seeds[@]}"; do
    output_prefix=outputxx/${model_name_or_path}/${tuning_method}/${method}_lr${lr}_${lr_type}/${seed}
    mkdir -p ${output_prefix}/order1/logs
    LOGFILE="${output_prefix}/order1/logs/train_and_infer_${ini_threshold}.log"
    # bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order_1 ${lr} ${lr_type} ${output_prefix}
    # bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order_1 1e-5 cosine
    # if [ ! -f "$LOGFILE" ]; then
    bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order1 ${lr} ${lr_type} ${output_prefix} > "$LOGFILE" 2>&1
    # else
    #     echo "Log file already exists: $LOGFILE"
    # fi

    mkdir -p ${output_prefix}/order2/logs
    LOGFILE="${output_prefix}/order2/logs/train_and_infer_${ini_threshold}.log"
    # if [ ! -f "$LOGFILE" ]; then
    bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order2 ${lr} ${lr_type} ${output_prefix} > "$LOGFILE" 2>&1
    # else
    #     echo "Log file already exists: $LOGFILE"
    # fi

    mkdir -p ${output_prefix}/order3/logs
    LOGFILE="${output_prefix}/order3/logs/train_and_infer_${ini_threshold}.log"
    # if [ ! -f "$LOGFILE" ]; then
    bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order3 ${lr} ${lr_type} ${output_prefix} > "$LOGFILE" 2>&1
    # else
    #     echo "Log file already exists: $LOGFILE"
    # fi
    mkdir -p ${output_prefix}/order4/logs
    LOGFILE="${output_prefix}/order4/logs/train_and_infer_${ini_threshold}.log"
    bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order4 ${lr} ${lr_type} ${output_prefix} > "$LOGFILE" 2>&1
    # else
    #     echo "Log file already exists: $LOGFILE"
    # fi

    mkdir -p ${output_prefix}/order5/logs
    LOGFILE="${output_prefix}/order5/logs/train_and_infer_${ini_threshold}.log"
    # if [ ! -f "$LOGFILE" ]; then
    bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order5 ${lr} ${lr_type} ${output_prefix} > "$LOGFILE" 2>&1
    # else
    #     echo "Log file already exists: $LOGFILE"
    # fi

    mkdir -p ${output_prefix}/order6/logs
    LOGFILE="${output_prefix}/order6/logs/train_and_infer_${ini_threshold}.log"
    # if [ ! -f "$LOGFILE" ]; then
    bash scripts/order.sh ${model} ${tuning_method} ${method} ${ini_threshold} ${seed} ${gpus} order6 ${lr} ${lr_type} ${output_prefix} > "$LOGFILE" 2>&1

  done
done