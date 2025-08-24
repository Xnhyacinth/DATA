


# nohup bash scripts/train.sh 0,1,2,3,4,5,6,7 llama2-7b 1e-4 constant lora_tuning olora > logs/llama2.log 2>&1 &

# nohup bash scripts/train1.sh 4,5,6,7 llama2-7b 1e-4 constant lora_tuning olora > logs/llama2.log 2>&1 &

nohup bash scripts/train.sh 1,3,6,7 t5-large 1e-3 constant lora_tuning olora > logs/t5.log 2>&1 &


# nohup bash scripts/train1.sh 0,1,2,5 llama3.1-8b 1e-4 constant lora_tuning olora > logs/llama3.log 2>&1 &