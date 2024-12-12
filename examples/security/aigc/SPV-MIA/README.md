# SPV-MIA

This repository aims to replicating the results in paper  [Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration](https://arxiv.org/abs/2311.06062). [^1]

[^1]: This repo is a replica of the [SPV-MIA's official implementation](https://github.com/tsinghua-fib-lab/ANeurIPS2024_SPV-MIA). We updated the fine-tuning implementations, such as the officially supported flash_attention_2 for llama by transformers, and cleaned up the code.

This paper proposes a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, they introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. Furthermore, they introduce probabilistic variation, a more reliable membership signal based on LLM memorization rather than overfitting, from which we rediscover the neighbour attack with theoretical grounding.

This repo is supported by [ANT group](https://www.antgroup.com/) and [Cybersecurity College Student Innovation Funding Program](https://zzjh.org.cn/#/)

## Setup

### Environment
The code is based on python 3.10. It is recommended to use conda create env:

```bash
conda create -n MIA python=3.10
conda activate MIA
```

### Requirements

Dependency can be installed with the following command:

```bash
pip install -r requirements.txt
```

## Tested pretrained models
- GPT-2 (https://huggingface.co/gpt2)
- GPT-J (https://huggingface.co/EleutherAI/gpt-j-6b)
- Falcon (https://huggingface.co/tiiuae/falcon-7b)
- LLaMA (https://huggingface.co/luodian/llama-7b-hf) [^2]

[^2]: This third-party repo `luodian/llama-7b-hf` seems to be deleted by unknown reasons, using forked repos [luodian/llama-7b-hf](https://huggingface.co/luodian/llama-7b-hf) or [baffo32/decapoda-research-llama-7B-hf](https://huggingface.co/baffo32/decapoda-research-llama-7B-hf) as alternatives.

## Tested datasets
- Ag News (https://huggingface.co/datasets/ag_news)
- Wikitext-103 (https://huggingface.co/datasets/wikitext) [^3]
- Xsum (https://huggingface.co/datasets/xsum)

[^3]: Please add an additional argument `--dataset_config_name wikitext-2-raw-v1` to specify this dataset.

## Target Model Fine-tuning
  All large language models (LLMs) are built on the top of [transformers](https://huggingface.co/docs/transformers/index), 
  a go-to library for state-of-the-art transformer models, on which you can fine-tune arbitrary well-known LLMs you want,
  including LLaMA, GPT-series, Falcon, etc.
  We recommend training LLMs with multi-GPU and [accelerate](https://huggingface.co/docs/accelerate/index), 
  a library that enables the same PyTorch code to be run across any distributed configuration:
  ```bash
accelerate launch ./ft_llms/llms_finetune.py \
--output_dir ./ft_llms/*pretrained_model_name*/*dataset_name*/target/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d *dataset_name* -m *pretrained_model_name* --packing --use_dataset_cache \
-e 10 -b 4 -lr 1e-4 --gradient_accumulation_steps 1 \
--train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000
  ```
Please replace \*pretrained_model_name\* and \*dataset_name\* with the names of pretrained LLM and training dataset, such as `luodian/llama-7b-hf` and `ag_news`.

## Reference dataset sampling
  Before fine-tuning the self-prompt reference model, the reference dataset can be sampled via the proposed self-prompt approach over the fine-tuned LLM. 
  ```bash
  accelerate launch refer_data_generate.py \
-tm *fine_tuned_model* \
-m *pretrained_model_name* -d *dataset_name*
  ```
  Replace \*fine_tuned_model\* with the directory of the fine-tuned target model (i.e., the output directory of 
  the [Target Model Fine-tuning](#target-model-fine-tuning) phase). 

## Self-prompt Reference Model Fine-tuning
 Then fine-tune the self-prompt reference model in the same manner as the target model, but with a smaller training epoch:
```bash
accelerate launch ./ft_llms/llms_finetune.py --refer \
--output_dir ./ft_llms/*pretrained_model_name*/*dataset_name*/refer/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d *dataset_name* -m *pretrained_model_name* --packing --use_dataset_cache \
-e 2 -b 4 -lr 5e-5 --gradient_accumulation_steps 1 \
--train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000
```


## Run SPV-MIA
After accomplishing the preliminary operations, here is the command for deploying SPV-MIA on the target model.
```bash
python attack.py -m *pretrained_model_name* \
-tm *target_fine_tuned_model_path* \
-rm *reference_fine_tuned_model_path* \
-d *dataset_name* 
```
  Replace \*pretrained_model_name\* and \*dataset_name\* with the names of pretrained LLM and training dataset, such as luodian/llama-7b-hf and ag_news. And Relace the \*target_fine_tuned_model_path\* and \*reference_fine_tuned_model_path\* with the directory of the fine-tuned target model path and refenence model path. 

## All in One Script
For convenience, an all-in-one script is provided to streamline the process. We tested the entire process on four pre-trained models in [Tested pretrained models](#tested-pretrained-models) and three datasets in [Tested datasets](#tested-datasets).

Run the all-in-one script:
```bash
bash run.sh
```
The attack log is redirected to attack_result.txt.