#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)

model=$1
tuning_method=$2
method=$3
ini_threshold=$4
seed=$5
gpus=$6
order_type=${7:-"order_1"}
lr=${8:-"1e-4"}
lr_type=${9:-"constant"}
output_prefix=${10:-"0"}
m="${model#*/}"
i=0
weights=tuning_weight
python_file=src/run_uie_ft.py
ds_config=stage2_llama.config
# ds_config=stage0.config
warmup_ratio=0.02
if [[ $tuning_method == *lora* ]]
then
   weights=adapter
   python_file=src/run_uie_lora.py
fi
if [[ $model == *t5* ]]
then
   ds_config=stage0.config
fi
if [ "$lr_type" == "constant" ];then
   warmup_ratio=0.00
fi

if [ "$order_type" == "order1" ];then
   orders=dbpedia,amazon,yahoo,agnews
fi
if [ "$order_type" == "order2" ];then
   orders=dbpedia,amazon,agnews,yahoo
fi
if [ "$order_type" == "order3" ];then
   orders=yahoo,amazon,agnews,dbpedia
fi
# if [ "$order_type" == "order4" ];then
#    orders=MNLI,CB,WiC,COPA,QQP,BoolQA,RTE,IMDB,yelp,amazon,SST-2,dbpedia,agnews,MultiRC,yahoo
# fi
# if [ "$order_type" == "order5" ];then
#    orders=MultiRC,BoolQA,WiC,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yelp,amazon,yahoo
# fi
# if [ "$order_type" == "order6" ];then
#    orders=yelp,amazon,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yahoo,MultiRC,BoolQA,WiC
# fi
if [ "$order_type" == "order4" ];then
   orders=mnli,cb,wic,copa,qqp,boolqa,rte,imdb,yelp,amazon,sst-2,dbpedia,agnews,multirc,yahoo
fi
if [ "$order_type" == "order5" ];then
   orders=multirc,boolqa,wic,mnli,cb,copa,qqp,rte,imdb,sst-2,dbpedia,agnews,yelp,amazon,yahoo
fi
if [ "$order_type" == "order6" ];then
   orders=yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst-2,dbpedia,agnews,yahoo,multirc,boolqa,wic
fi
if [ "$output_prefix" == "0" ];then
   output_prefix=output/${m}/${tuning_method}/${method}_lr${lr}_${lr_type}/${seed}
fi
last_element=$(echo $orders | awk -F ',' '{print $NF}')
IFS=',' read -r -a parts <<< "$orders"
echo ${orders}
extra_args=""
flag=1
for part in "${parts[@]}"; do
    echo "$part"
    if [ "$i" != "0" ];then
        model=${output_prefix}/${order_type}/outputs/${i}-${pre_part}/${weights}
    fi
    pre_part=${part}
    ((i+=1))
   #  if [ "$part" == "rte" ];then
   #      flag=0
   #  fi
   #  if [ "$flag" == "1" ];then
   #      continue
   #  fi
    echo "model: ${model}"
    echo "output_dir: ${output_prefix}/${order_type}/outputs/${i}-${part}"
    # CUDA_VISIBLE_DEVICES=${gpus} torchrun --master_port=$port --nproc_per_node=2 ${python_file} \
    if [ "$part" == "$last_element" ];then
        extra_args="${extra_args}  --do_predict --predict_with_generate"
    fi
    deepspeed --master_port $port --include localhost:${gpus} ${python_file} \
      --do_train \
      --model_name_or_path ${model} \
      --data_dir CL_Benchmark \
      --task_config_dir configs/${order_type}_configs/${part} \
      --instruction_file configs/instruction_config.json \
      --instruction_strategy single \
      --output_dir ${output_prefix}/${order_type}/outputs/${i}-${part} \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 8 \
      --learning_rate ${lr} \
      --num_train_epochs 1 \
      --run_name ${order_type}_round${i} \
      --max_source_length 450 \
      --max_target_length 50 \
      --generation_max_length 50 \
      --add_task_name True \
      --add_dataset_name True \
      --overwrite_output_dir \
      --overwrite_cache \
      --lr_scheduler_type ${lr_type} \
      --warmup_ratio ${warmup_ratio} \
      --logging_strategy steps \
      --logging_steps 10 \
      --evaluation_strategy no \
      --save_strategy no \
      --save_steps 1500 \
      --lamda_1 0.5 \
      --lamda_2 0 \
      --seed ${seed} \
      --deepspeed configs/ds_configs/${ds_config} \
      --bf16 \
      ${extra_args}
      # --fp16
      # --bf16 

   sleep 15
done

