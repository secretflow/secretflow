#!/bin/bash

model_id="stable-diffusion-v1-5/stable-diffusion-v1-5"
img_num=20

datasets=("celebahq-caption" "wikiart")

checkpoint_suffixes=(
    "clean" 
    "advdm" 
    "antidb"
    # DiffPure
    "advdm_diffpure_steps50"
    "advdm_diffpure_steps100"
    "antidb_diffpure_steps50"
    "antidb_diffpure_steps100"
    # GridPure
    "advdm_gridpure"
    "antidb_gridpure"
)


text_encoder_options=(0 1)


mkdir -p output

for dataset in "${datasets[@]}"; do
    if [ "$dataset" = "wikiart" ]; then
        prompt="a painting in the style of Pablo Picasso"
    else
        prompt="a photography of a woman with long blonde hair and blue eyes"
    fi
    
    for checkpoint_suffix in "${checkpoint_suffixes[@]}"; do
        for text_encoder_option in "${text_encoder_options[@]}"; do
            if [[ $checkpoint_suffix == *"diffpure"* ]]; then
                base_suffix=$(echo $checkpoint_suffix | sed 's/_steps[0-9]*//')
                steps=$(echo $checkpoint_suffix | grep -o 'steps[0-9]*')
                
                if [ "$text_encoder_option" = "1" ]; then
                    checkpoint_dir="./checkpoint/${dataset}_${base_suffix}_with_text_encoder_${steps}"
                    output_subdir="output/${dataset}_${base_suffix}_with_text_encoder_${steps}"
                else
                    checkpoint_dir="./checkpoint/${dataset}_${checkpoint_suffix}"
                    output_subdir="output/${dataset}_${checkpoint_suffix}"
                fi
            else
                if [ "$text_encoder_option" = "1" ]; then
                    checkpoint_dir="./checkpoint/${dataset}_${checkpoint_suffix}_with_text_encoder"
                    output_subdir="output/${dataset}_${checkpoint_suffix}_with_text_encoder"
                else
                    checkpoint_dir="./checkpoint/${dataset}_${checkpoint_suffix}"
                    output_subdir="output/${dataset}_${checkpoint_suffix}"
                fi
            fi
            
            if [ ! -d "$checkpoint_dir" ]; then
                echo "Warning: Checkpoint directory $checkpoint_dir does not exist. Skipping..."
                continue
            fi

            if [ -d "$output_subdir" ] && [ "$(ls -A $output_subdir)" ]; then
                echo "Output directory $output_subdir already exists and is not empty. Skipping..."
                continue
            fi

            mkdir -p "$output_subdir"
            
            echo "==============================================="
            echo "Generating images for configuration:"
            echo "Dataset: $dataset"
            echo "Checkpoint: $checkpoint_dir"
            echo "Text encoder: $text_encoder_option"
            echo "Output directory: $output_subdir"
            
            CUDA_VISIBLE_DEVICES=0 python generate.py \
                --model_id="$model_id" \
                --lora_dir="$checkpoint_dir" \
                --output_dir="$output_subdir" \
                --prompt="$prompt" \
                --img_num=$img_num \
                --train_text_encoder=$text_encoder_option

            if [ $? -eq 0 ]; then
                echo "Successfully generated images for current configuration"
            else
                echo "Failed to generate images for current configuration"
            fi
            
            echo "-----------------------------------------------"
        done
    done
done

echo "Generation process completed!"