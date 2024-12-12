#!/bin/bash
set -e

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=36000


models=("luodian/llama-7b-hf" "tiiuae/falcon-7b" "EleutherAI/gpt-j-6b" "gpt2")
datasets=("ag_news" "wikitext" "xsum")
block_size=128
eval_steps=500
save_epochs=500
log_steps=500
train_sta_idx=0
train_end_idx=10000
prompt_sta_idx=10000
prompt_end_idx=20000
eval_sta_idx=0
eval_end_idx=1000
epochs_target=10
epochs_refer=4
batch_size=4
lr_target=1e-4
lr_refer=5e-5

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do

    dataset_config=""
    peft_config=""
    if [[ "$dataset" == "wikitext" ]]; then
      dataset_config="--dataset_config_name wikitext-2-raw-v1"
    fi
    # if model is gpt2, --disable_peft is needed
    if [[ "$model" == "gpt2" ]]; then
      peft_config="--disable_peft"
    fi

    echo "Processing model: $model, dataset: $dataset"

    # Target Model Fine-tuning
    echo "Starting Target Model Fine-tuning for $model on $dataset"
    accelerate launch ./ft_llms/llms_finetune.py \
      --output_dir ./ft_llms/$model/$dataset/target/ \
      --block_size $block_size --eval_steps $eval_steps --save_epochs $save_epochs --log_steps $log_steps \
      -d $dataset -m $model --packing --use_dataset_cache \
      -e $epochs_target -b $batch_size -lr $lr_target --gradient_accumulation_steps 1 \
      --train_sta_idx=$train_sta_idx --train_end_idx=$train_end_idx --eval_sta_idx=$eval_sta_idx --eval_end_idx=$eval_end_idx \
      $dataset_config $peft_config

    # Generate Reference Data
    echo "Generating Reference Data for $model on $dataset"
    accelerate launch ./ft_llms/refer_data_generate.py \
      -tm ./ft_llms/$model/$dataset/target/  \
      -m $model -d $dataset --prompt_sta_idx=$prompt_sta_idx --prompt_end_idx=$prompt_end_idx \
      $dataset_config

    # Self-prompt Reference Model Fine-tuning
    echo "Starting Reference Model Fine-tuning for $model on $dataset"
    accelerate launch ./ft_llms/llms_finetune.py --refer \
      --output_dir ./ft_llms/$model/$dataset/refer/ \
      --block_size $block_size --eval_steps $eval_steps --save_epochs $save_epochs --log_steps $log_steps \
      -d $dataset -m $model --packing --use_dataset_cache  \
      -e $epochs_refer -b $batch_size -lr $lr_refer --gradient_accumulation_steps 1 \
      --train_sta_idx=$train_sta_idx --train_end_idx=$train_end_idx --eval_sta_idx=$eval_sta_idx --eval_end_idx=$eval_end_idx \
      $dataset_config $peft_config

    # Attack
    echo "Running Attack for $model on $dataset" 
    python attack.py -m $model \
      -tm ./ft_llms/$model/$dataset/target/ \
      -rm ./ft_llms/$model/$dataset/refer/ \
      -d $dataset $dataset_config
  done
done

echo "All tasks completed."
