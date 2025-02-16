seeds=(42)
num_gpus=${1:-"1"}
gpus=${2:-"1"}
model=${3:-"t5-large"}
tuning_method=${4:-"lora"}
bs=${5:-"8"}
lr_type=${6:-"constant"}
lr=${7:-"1e-4"}
filter=${8:-"0"}
mode=${9:-"all"}
select=${10:-"0"}
r=${11:-"8"}
deepspeed=${12:-"-1"}
vida_rank1=${13:-"4"}
vida_rank2=${14:-"16"}
restore=${15:-"0"}
scale=${16:-"0"}
adaprompt=${17:-"0"}
reinit=${18:-"0"}
ortho_mu=${19:-"0"}
gap_layers=${20:-"4"}
bakebone=${21-"0"}
nomlp=${22:-"0"}
project=${23:-"0"}
replay=${24:-"0"}
# bash config/train.sh 1 9 tinyllama order_1 lora 1 1e-4 8 fewshot -1 1e-10 8
for i in {4..6}; do
    for seed in "${seeds[@]}"; do
        output_prefix=logs/${model}/${tuning_method}/${seed}/order_${i}/${lr_type}_${lr}_${bs}
        if [ "$filter" != "0" ];then
            output_prefix="${output_prefix}_${filter}"
        fi
        if [ "$select" != "0" ];then
            output_prefix="${output_prefix}_${select}"
        fi
        if [ "$mode" != "all" ];then
            output_prefix="${output_prefix}_${mode}"
        fi
        if [ "$tuning_method" == "lora" ];then
            output_prefix="${output_prefix}_r${r}"
        fi
        if [ "$tuning_method" == "vida" ];then
            output_prefix="${output_prefix}_vida${vida_rank1}_vida${vida_rank2}"
        fi
        if [ "$deepspeed" != "-1" ];then
            output_prefix="${output_prefix}_ds${deepspeed}"
        fi
        if [ "$restore" != "0" ];then
            output_prefix="${output_prefix}_restore${restore}"
        fi
        if [ "$scale" != "0" ];then
            output_prefix="${output_prefix}_scale${scale}"
        fi
        if [ "$adaprompt" != "0" ];then
            output_prefix="${output_prefix}_adaprompt${adaprompt}"
        fi
        if [ "$reinit" != "0" ];then
            output_prefix="${output_prefix}_reinit"
        fi
        if [ "$ortho_mu" != "0" ];then
            output_prefix="${output_prefix}_ortho_mu${ortho_mu}"
        fi
        if [ "$gap_layers" != "4" ];then
            output_prefix="${output_prefix}_gap_layers${gap_layers}"
        fi
        if [ "$bakebone" != "0" ];then
            output_prefix="${output_prefix}_bakebone"
        fi
        if [ "$nomlp" != "0" ];then
            output_prefix="${output_prefix}_nomlp"
        fi
        if [ "$project" != "0" ];then
            output_prefix="${output_prefix}_project"
        fi
        if [ "$replay" != "0" ];then
            output_prefix="${output_prefix}_replay"
        fi
        mkdir -p ${output_prefix}
        LOGFILE="${output_prefix}/train_and_infer.log"
        # bash config/train4.sh ${num_gpus} ${gpus} ${model} order_${i} ${tuning_method} 1 ${lr} ${bs} fewshot ${deepspeed} 1e-10 ${r} ${lr_type} ${seed} ${filter} ${mode} ${select} ${vida_rank1} ${vida_rank2} ${restore} ${scale} ${adaprompt} ${reinit} ${ortho_mu} ${gap_layers} ${bakebone} ${nomlp} ${project} ${replay}
        bash config/train.sh ${num_gpus} ${gpus} ${model} order_${i} ${tuning_method} 1 ${lr} ${bs} fewshot ${deepspeed} 1e-10 ${r} ${lr_type} ${seed} ${filter} ${mode} ${select} ${vida_rank1} ${vida_rank2} ${restore} ${scale} ${adaprompt} ${reinit} ${ortho_mu} ${gap_layers} ${bakebone} ${nomlp} ${project} ${replay} > "$LOGFILE" 2>&1
    done
done
# 