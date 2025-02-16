

export WANDB_API_KEY=
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
model=${3:-"llama3-8b"}
order=${4:-"order_1"}
finetuning_type=${5:-"lora"}
epoch=${6:-"1"}
lr=${7:-"1e-4"}
bs=${8:-"8"}
template=${9:-"fewshot"}
deepspeed=${10:-"-1"}
eval=${11:-"1e-10"}
r=${12:-"8"}
lr_scheduler_type=${13:-"constant"}
seed=${14:-"42"}
filter=${15:-"0"}
mode=${16:-"all"}
select=${17:-"0"}
vida_rank1=${18:-"4"}
vida_rank2=${19:-"16"}
restore=${20:-"0"}
scale=${21:-"0"}
adaprompt=${22:-"0"}
reinit=${23:-"0"}
ortho_mu=${24:-"0"}
gap_layers=${25:-"4"}
bakebone=${26:-"0"}
nomlp=${27:-"0"}
project=${28:-"0"}
replay=${29:-"0"}
max_samples=${30:-"1000000"}
extra_args=""
save_steps=10000
cutoff_len=2048
gradient_accumulation_steps=1
warmup_ratio=0.05
max_new_tokens=50
# bash patch/install.sh
random_port=$((RANDOM%(65535-1024+1)+1024))
while [[ $(ss -tln | grep ":$random_port") ]]; do
    random_port=$((RANDOM%(65535-1024+1)+1024))
done
export MASTER_PORT=${random_port}
if [ "$lr_scheduler_type" == "constant" ];then
   warmup_ratio=0.00
fi

if [ "$order" == "order_1" ];then
   orders=dbpedia,amazon,yahoo,agnews
fi
if [ "$order" == "order_2" ];then
   orders=dbpedia,amazon,agnews,yahoo
fi
if [ "$order" == "order_3" ];then
   orders=yahoo,amazon,agnews,dbpedia
fi
if [ "$order" == "order_4" ];then
   orders=mnli,cb,wic,copa,qqp,boolqa,rte,imdb,yelp,amazon,sst-2,dbpedia,agnews,multirc,yahoo
fi
if [ "$order" == "order_5" ];then
   orders=multirc,boolqa,wic,mnli,cb,copa,qqp,rte,imdb,sst-2,dbpedia,agnews,yelp,amazon,yahoo
fi
if [ "$order" == "order_6" ];then
   orders=yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst-2,dbpedia,agnews,yahoo,multirc,boolqa,wic
fi
# trace
if [ "$order" == "order_7" ];then
   orders=c-stance,fomc,meetingbank,py150,scienceqa,numglue-cm,numglue-ds,20minuten
fi
last_element=$(echo $orders | awk -F ',' '{print $NF}')
IFS=',' read -r -a parts <<< "$orders"
orders=${orders//,/ }
echo ${orders}


if [ "$bs" = "4" ];then
    gradient_accumulation_steps=2
fi
if [ "$bs" = "2" ];then
    gradient_accumulation_steps=4
fi
if [ "$bs" = "1" ];then
    gradient_accumulation_steps=8
fi
if [ "$num_gpus" == "1" ] && [ "$bs" != "16" ];then
    gradient_accumulation_steps=$((2 * gradient_accumulation_steps))
fi

let eval_bs=bs*4
eval_bs=8
if [ "$eval_bs" -gt 16 ]; then
    eval_bs=16
fi
echo ""
model_name_or_path=${model}
model="${model_name_or_path##*/}"
if [ "$model" = "llama3-8b" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B
    cutoff_len=4096
fi
if [ "$model" = "llama3.1-8b" ];then
    model_name_or_path=meta-llama/Llama-3.1-8B
    cutoff_len=1024
    # gradient_accumulation_steps=$((gradient_accumulation_steps / 2))
    if [ "$adaprompt" != "0" ] && [ "$restore" != "0" ];then
        cutoff_len=800
    fi
fi
if [ "$model" = "llama3-8b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
    template=llama3
fi
if [ "$model" = "llama2-7b" ];then
    model_name_or_path=meta-llama/Llama-2-7b-hf
    cutoff_len=1024
    if [ "$adaprompt" != "0" ] && [ "$restore" != "0" ];then
        cutoff_len=800
    fi
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-0.5b" ];then
    model_name_or_path=Qwen/Qwen2-0.5B
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-1.5b" ];then
    model_name_or_path=Qwen/Qwen2-1.5B
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-7b" ];then
    model_name_or_path=Qwen/Qwen2-7B
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "llama2-7b-chat" ];then
    model_name_or_path=meta-llama/Llama-2-7b-chat-hf
    template=llama2
fi
if [ "$model" = "tinyllama" ];then
    model_name_or_path=TinyLlama/TinyLlama_v1.1
fi
if [ "$model" = "tinyllama-chat" ];then
    model_name_or_path=TinyLlama/TinyLlama-1.1B-Chat-v1.0
fi
if [ "$model" = "llama3-70b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-70B-Instruct
    template=llama3
fi
if [ "$model" = "qwen2.5-1.5b" ];then
    model_name_or_path=Qwen/Qwen2.5-1.5B
    cutoff_len=4096
fi
if [ "$model" = "qwen2.5-3b" ];then
    model_name_or_path=Qwen/Qwen2.5-3B
    cutoff_len=4096
fi
if [ "$model" = "qwen2.5-7b" ];then
    model_name_or_path=Qwen/Qwen2.5-7B
    cutoff_len=4096
fi
if [ "$model" = "t5-large" ];then
    model_name_or_path=google-t5/t5-large
    template=t5
    cutoff_len=1024
    if [ "$adaprompt" != "0" ] && [ "$restore" != "0" ];then
        cutoff_len=800
    fi
fi
# extra_args="${extra_args} --disable_gradient_checkpointing True"
save_path=saves/${model}/${finetuning_type}/sft/${order}/${seed}/${lr_scheduler_type}
run_name=LLM/${model}/${finetuning_type}/sft/${order}/${seed}/${lr_scheduler_type}
merge_path=models/${model}_${finetuning_type}_sft_${order}/${seed}/${lr_scheduler_type}
eval_path=saves/${model}/${finetuning_type}/eval/${order}/${seed}/${lr_scheduler_type}

# save_path=${save_path//\,/_}
# run_name=${run_name//\,/_}
# merge_path=${merge_path//\,/_}
# eval_path=${eval_path//\,/_}

if [ "$finetuning_type" = "lora" ];then
    lora_rank=${r}
    lora_dropout=0.1
    lora_target=all
    # lora_alpha=16
    # "Name(s) of target modules to apply LoRA. "
    # "Use commas to separate multiple modules. "
    # "Use `all` to specify all the linear modules. "
    # "LLaMA choices: [`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], "
    # "BLOOM & Falcon & ChatGLM choices: [`query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`], "
    # "Baichuan choices: [`W_pack`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], "
    # "Qwen choices: [`c_attn`, `attn.c_proj`, `w1`, `w2`, `mlp.c_proj`], "
    # "InternLM2 choices: [`wqkv`, `wo`, `w1`, `w2`, `w3`], "
    # "Others choices: the same as LLaMA."
    extra_args="$extra_args --lora_rank ${lora_rank} --lora_dropout ${lora_dropout} --lora_target ${lora_target}"
    if [ "$lora_rank" != "8" ];then
        save_path="${save_path}_r${lora_rank}"
        run_name="${run_name}_r${lora_rank}"
        merge_path="${merge_path}_r${lora_rank}"
        eval_path="${eval_path}_r${lora_rank}"
    fi
fi

if [ "$finetuning_type" == "vida" ];then
    extra_args="$extra_args --vida_rank1 ${vida_rank1} --vida_rank2 ${vida_rank2} --is_vida True"
    # if [ "$vida_rank1" != "4" ];then
    save_path="${save_path}_vida${vida_rank1}_vida${vida_rank2}"
    run_name="${run_name}_vida${vida_rank1}_vida${vida_rank2}"
    merge_path="${merge_path}_vida${vida_rank1}_vida${vida_rank2}"
    eval_path="${eval_path}_vida${vida_rank1}_vida${vida_rank2}"
    # fi
    finetuning_type=full
    is_vida=True
fi

if [ "$deepspeed" != "-1" ];then
    extra_args="$extra_args --deepspeed examples/deepspeed/ds_z${deepspeed}_config.json"
    save_path="${save_path}_ds${deepspeed}"
    run_name="${run_name}_ds${deepspeed}"
    merge_path="${merge_path}_ds${deepspeed}"
    eval_path="${eval_path}_ds${deepspeed}"
fi

if [ "$eval" != "1e-10" ];then
    extra_args="$extra_args --per_device_eval_batch_size ${eval_bs} --val_size ${eval} --eval_steps ${save_steps} --evaluation_strategy steps --load_best_model_at_end"
    save_path="${save_path}_eval"
    run_name="${run_name}_eval"
    merge_path="${merge_path}_eval"
    eval_path="${eval_path}_eval"
fi

if [ "$lr" != "1e-4" ];then
    save_path="${save_path}_lr${lr}"
    run_name="${run_name}_lr${lr}"
    merge_path="${merge_path}_lr${lr}"
    eval_path="${eval_path}_lr${lr}"
fi

if [ "$epoch" != "1" ];then
    save_path="${save_path}_epoch${epoch}"
    run_name="${run_name}_epoch${epoch}"
    merge_path="${merge_path}_epoch${epoch}"
    eval_path="${eval_path}_epoch${epoch}"
fi

if [ "$max_samples" != "1000000" ];then
    save_path="${save_path}_${max_samples}"
    run_name="${run_name}_${max_samples}"
    merge_path="${merge_path}_${max_samples}"
    eval_path="${eval_path}_${max_samples}"
fi

if [ "$filter" != "0" ];then
    save_path="${save_path}_${filter}"
    run_name="${run_name}_${filter}"
    merge_path="${merge_path}_${filter}"
    eval_path="${eval_path}_${filter}"
fi

if [ "$select" != "0" ];then
    save_path="${save_path}_${select}"
    run_name="${run_name}_${select}"
    merge_path="${merge_path}_${select}"
    eval_path="${eval_path}_${select}"
    extra_args="${extra_args} --select ${select}"
fi

if [ "$scale" != "0" ];then
    extra_args="$extra_args --scale ${scale}"
    save_path="${save_path}_scale${scale}"
    run_name="${run_name}_scale${scale}"
    merge_path="${merge_path}_scale${scale}"
    eval_path="${eval_path}_scale${scale}"
fi

if [ "$restore" != "0" ];then
    extra_args="$extra_args --restore ${restore}"
    save_path="${save_path}_restore"
    run_name="${run_name}_restore"
    merge_path="${merge_path}_restore"
    eval_path="${eval_path}_restore"
    if [ "$restore" != "1" ];then
        save_path="${save_path}${restore}"
        run_name="${run_name}${restore}"
        merge_path="${merge_path}${restore}"
        eval_path="${eval_path}${restore}"
    fi
fi
#  extra_args="${extra_args} --disable_gradient_checkpointing True"
if [ "$adaprompt" != "0" ];then
    extra_args="$extra_args --adaprompt ${adaprompt}  --disable_gradient_checkpointing True"
    save_path="${save_path}_adaprompt${adaprompt}"
    run_name="${run_name}_adaprompt${adaprompt}"
    merge_path="${merge_path}_adaprompt${adaprompt}"
    eval_path="${eval_path}_adaprompt${adaprompt}"
fi

if [ "$reinit" != "0" ];then
    extra_args="$extra_args --reinit True"
    save_path="${save_path}_reinit${reinit}"
    run_name="${run_name}_reinit${reinit}"
    merge_path="${merge_path}_reinit${reinit}"
    eval_path="${eval_path}_reinit${reinit}"
fi

if [ "$ortho_mu" != "0" ];then
    extra_args="$extra_args --ortho_mu ${ortho_mu}"
    save_path="${save_path}_ortho_mu${ortho_mu}"
    run_name="${run_name}_ortho_mu${ortho_mu}"
    merge_path="${merge_path}_ortho_mu${ortho_mu}"
    eval_path="${eval_path}_ortho_mu${ortho_mu}"
fi

if [ "$gap_layers" != "4" ];then
    extra_args="$extra_args --gap_layers ${gap_layers}"
    save_path="${save_path}_gap_layers${gap_layers}"
    run_name="${run_name}_gap_layers${gap_layers}"
    merge_path="${merge_path}_gap_layers${gap_layers}"
    eval_path="${eval_path}_gap_layers${gap_layers}"
fi

if [ "$bakebone" != "0" ];then
    extra_args="$extra_args --scale_bakebone"
    save_path="${save_path}_bakebone"
    run_name="${run_name}_bakebone"
    merge_path="${merge_path}_bakebone"
    eval_path="${eval_path}_bakebone"
fi

if [ "$nomlp" != "0" ];then
    extra_args="$extra_args --nomlp"
    save_path="${save_path}_nomlp"
    run_name="${run_name}_nomlp"
    merge_path="${merge_path}_nomlp"
    eval_path="${eval_path}_nomlp"
fi

if [ "$project" != "0" ];then
    extra_args="$extra_args --project ${project}"
    save_path="${save_path}_project${project}"
    run_name="${run_name}_project${project}"
    merge_path="${merge_path}_project${project}"
    eval_path="${eval_path}_project${project}"
fi

if [ "$replay" != "0" ];then
    extra_args="$extra_args --replay"
    save_path="${save_path}_replay"
    run_name="${run_name}_replay"
    merge_path="${merge_path}_replay"
    eval_path="${eval_path}_replay"
fi

if [ "$mode" == "all" ];then
    # extra_args="${extra_args} --do_train --do_predict --predict_with_generate"
    extra_args="${extra_args} --do_train"
fi

if [ "$mode" == "eval" ];then
    extra_args="${extra_args} --do_predict --predict_with_generate"
    model_prefix=${save_path}
    save_path="${save_path}_${mode}"
    run_name="${run_name}_${mode}"
    merge_path="${merge_path}_${mode}"
    eval_path="${eval_path}_${mode}"
fi

if [[ $model_name_or_path != *t5* ]]; then
    extra_args="${extra_args} --flash_attn fa2"
fi

save_prefix=${save_path}
extra_args0=${extra_args}
# Train
idx=0
bs0=${bs}
cutoff_len0=${cutoff_len}
gradient_accumulation_steps0=${gradient_accumulation_steps}
flag=1
for part in "${parts[@]}"; do
    echo ""
    for (( i=1; i<=50; i++ ))
    do
        echo -n "*"
    done
    echo ""
    echo "$part"
    
    if [ "$idx" == "0" ];then
        eval_dataset="cl_${part}_eval"
        train_dataset=cl_${part}
    fi
    if [ "$idx" != "0" ];then
        eval_dataset="${eval_dataset},cl_${part}_eval"
        train_dataset="${train_dataset},cl_${part}"
        if [ "$finetuning_type" == "lora" ];then
            adapter_name_or_path=${save_prefix}/${idx}-${pre_part}
            extra_args="${extra_args0} --adapter_name_or_path ${adapter_name_or_path}"
        fi
        if [ "$is_vida" == "True" ];then
            model_name_or_path=${save_prefix}/${idx}-${pre_part}
            extra_args="${extra_args0}"
        fi
    fi
    mvpath="${save_prefix}/${idx}-${pre_part}"
    pre_part=${part}
    dataset=cl_${part}
    if [ "$filter" != "0" ];then
        dataset="${dataset}_${filter}"
    fi
    if [ "$replay" != "0" ];then
        dataset=${train_dataset}
    fi
    
    ((idx+=1))
    if [ "$mode" == "all" ];then
        extra_args="${extra_args} --dataset ${dataset} --eval_dataset ${eval_dataset}"
    fi
    if [ "$mode" == "eval" ];then
        if [ "$finetuning_type" == "lora" ];then
            extra_args="${extra_args0} --adapter_name_or_path ${model_prefix}/${idx}-${part}"
            extra_args="${extra_args} --eval_dataset ${eval_dataset}"
        fi
        if [ "$finetuning_type" == "full" ];then
            model_name_or_path=meta-llama/Llama-2-7b-hf
            extra_args="${extra_args0} --eval_dataset ${eval_dataset}"
        fi
        if [ "$is_vida" == "True" ];then
            model_name_or_path=${model_prefix}/${idx}-${part}
            extra_args="${extra_args0} --eval_dataset ${eval_dataset}"
        fi
    fi
    save_path="${save_prefix}/${idx}-${part}"
    if [ "$adaprompt" != "0" ];then
        task_id=$((idx-1))
        extra_args="${extra_args} --task_id ${task_id}"
    fi
    
    if [ "$is_vida" == "True" ];then
        if [ "$part" == "yahoo" ];then
            if [ "$bs" -gt 1 ]; then
                bs=$((bs0 / 2))
                gradient_accumulation_steps=$((gradient_accumulation_steps0 * 2))
            fi
        else
            bs=${bs0}
            gradient_accumulation_steps=${gradient_accumulation_steps0}
        fi
        # if [ "$part" != "yahoo" ] && [ "$part" != "dbpedia" ];then
        #     bs=${bs0}
        #     # cutoff_len=${cutoff_len0}
        #     gradient_accumulation_steps=${gradient_accumulation_steps0}
        # fi
        if [ "$adaprompt" != "0" ] && [ "$model" == "t5-large" ];then
            if [ "$part" == "dbpedia" ] || [ "$part" == "yelp" ] || [ "$part" == "multirc" ] || [ "$part" == "boolqa" ] || [ "$part" == "imdb" ];then
                if [ "$bs" -gt 1 ]; then
                    bs=$((bs0 / 2))
                    gradient_accumulation_steps=$((gradient_accumulation_steps0 * 2))
                fi
                # cutoff_len=1024
                # flag=0
            fi
        fi
    fi
    if [ "$part" == "$last_element" ];then
        extra_args="${extra_args}  --do_predict --predict_with_generate"
    fi
    # if [ "$flag" == "1" ];then
    #     continue
    # fi
    if [ "$part" == "c-stance" ] || [ "$part" == "py150" ] || [ "$part" == "numglue-cm" ] || [ "$part" == "numglue-ds" ];then
        epoch=5
    fi
    if [ "$part" == "meetingbank" ] || [ "$part" == "20minuten" ];then
        epoch=7
    fi
    if [ "$part" == "fomc" ] || [ "$part" == "scienceqa" ];then
        epoch=3
    fi
    if [ "$part" == "meetingbank" ] || [ "$part" == "scienceqa" ]|| [ "$part" == "20minuten" ];then
        max_new_tokens=512
    fi
    if [ "$part" == "meetingbank" ] && [ "$finetuning_type" == "lora" ];then
        bs=$((bs0 / 4))
    fi

    echo "model_name_or_path: ${model_name_or_path}"
    echo "template: ${template}"
    echo "save_path: ${save_path}"
    echo "bs: ${bs}"
    echo "cutoff_len: ${cutoff_len}"
    echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"
    echo "extra_args: ${extra_args}"
    echo "dataset: ${dataset}"
    echo "eval_dataset: ${eval_dataset}"

    succ=`CUDA_VISIBLE_DEVICES=${gpus} llamafactory-cli train \
        --stage cl \
        --model_name_or_path ${model_name_or_path} \
        --dataset_dir ./data \
        --template ${template} \
        --finetuning_type ${finetuning_type} \
        --output_dir ${save_path} \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len ${cutoff_len} \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size ${bs} \
        --per_device_eval_batch_size ${eval_bs} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --lr_scheduler_type ${lr_scheduler_type} \
        --logging_steps 10 \
        --warmup_ratio ${warmup_ratio} \
        --save_steps ${save_steps} \
        --save_total_limit 1 \
        --learning_rate ${lr} \
        --num_train_epochs ${epoch} \
        --max_samples ${max_samples} \
        --ddp_timeout 180000000 \
        --max_new_tokens ${max_new_tokens} \
        --plot_loss \
        --report_to wandb \
        --remove_unused_columns False \
        --run_name ${run_name} \
        --seed ${seed} \
        --bf16 \
        --orders ${orders} \
        ${extra_args} && echo "true" || echo "false"`
    sleep 20
    bash config/rm.sh ${save_path} checkpoint
    if [[ $succ != *true* ]]; then
        echo "${part} error!"
        exit 1
    fi
    if [ "$idx" -gt 1 ]; then
        echo "mv  ${mvpath}"
        bash config/rm.sh ${mvpath} safetensors
        sleep 20
    fi
done

# echo "mv  ${save_path}"
# bash config/rm.sh ${save_path} safetensors
sleep 30
