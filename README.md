# DATA

## 🛠 Requirements

Install LLaMA-Factory following [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
cd DATA
pip install -e ".[torch,metrics]"
```

## 💡 Data

- `data` folder has 15 tasks of Long Sequence Benchmark.

- Download the datasets from official websites.

- From Google drive: (we unified the formats of the above datasets). [Link]()


## Run

```shell
bash config/run1.sh ${num_gpus} ${gpus} ${model} ${tuning_method} ${bs} ${lr_type} ${lr} ${filter} ${mode} ${select} ${r} ${deepspeed} ${data_rank1} ${data_rank2} ${restore} ${scale} ${adaprompt} ${reinit} ${ortho_mu} ${gap_layers} ${bakebone} ${nomlp} ${project} ${replay}
```

### LLaMA2-7B

#### LoRA

```shell
bash config/run1.sh 2 0,1 llama2-7b lora 16 constant 1e-4 0 all 0 8 -1 0 0 0 0 0 0 0 0 0 0 0 0
```

#### LoRAReplay

```shell
bash config/run1.sh 2 0,1 llama2-7b lora 16 constant 1e-4 0 all 0 8 -1 0 0 0 0 0 0 0 0 0 0 0 1
```

#### DATA

```shell
bash config/run5.sh 2 0,1 llama2-7b data 1 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 8 1 1 4 0 0 0 0
```

### Details

Coming Soon!

## 🤝 Referencing and Citing 

If you find our work useful in your research and would like to cite our project, please use the following citation: found this work useful, please consider giving this repository a star and citing our paper as follows:
```bibtex
@article{liao2025data,
	title={DATA: Decomposed Attention-based Task Adaptation for Rehearsal-Free Continual Learning},
	author={Liao, Huanxuan and He, Shizhu and Hao, Yupu and Zhao, Jun and Liu, Kang },
	journal={arXiv preprint arXiv:2502.11482},
	year={2025}
}
```
