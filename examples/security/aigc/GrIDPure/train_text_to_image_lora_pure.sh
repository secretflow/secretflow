#!/bin/bash

datasets=("wikiart" "celebahq-caption")

data_roots=(
    "./purified_diffpure/advdm_data" 
    "./purified_diffpure/antidb_data"
    "./purified_gridpure/advdm_data"
    "./purified_gridpure/antidb_data"
)

text_encoder_options=("" "--train_text_encoder")

checkpoint_suffixes=(
    "advdm_diffpure" 
    "antidb_diffpure"
    "advdm_gridpure"
    "antidb_gridpure"
)

for dataset in "${datasets[@]}"; do
    if [ "$dataset" = "wikiart" ]; then
        instance_prompt="a painting in the style of Pablo Picasso"
    else
        instance_prompt="a photography of a woman with long blonde hair and blue eyes"
    fi

    for i in "${!data_roots[@]}"; do
        data_root="${data_roots[$i]}"
        checkpoint_suffix="${checkpoint_suffixes[$i]}"

        for text_encoder_option in "${text_encoder_options[@]}"; do

            if [ -n "$text_encoder_option" ]; then
                output_suffix="${checkpoint_suffix}_with_text_encoder"
            else
                output_suffix="${checkpoint_suffix}"
            fi
            

            if [[ $data_root == *"purified_diffpure"* ]]; then
                for steps in 50 100; do
                    echo "Running training for configuration:"
                    echo "Dataset: $dataset"
                    echo "Data root: $data_root"
                    echo "Steps: $steps"
                    echo "Text encoder: ${text_encoder_option:-disabled}"

                    current_data_dir="${data_root}/${dataset}_steps${steps}"
                    current_output_suffix="${output_suffix}_steps${steps}"
                    
                    CUDA_VISIBLE_DEVICES=0 accelerate launch train_text_to_image_lora.py \
                        --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
                        --instance_data_dir="$current_data_dir" \
                        --output_dir="./checkpoint/${dataset}_${current_output_suffix}" \
                        --instance_prompt="$instance_prompt" \
                        --resolution=512 \
                        --train_batch_size=2 \
                        --gradient_accumulation_steps=1 \
                        --checkpointing_steps=1000 \
                        --learning_rate=1e-4 \
                        --lr_scheduler="constant" \
                        --lr_warmup_steps=0 \
                        --max_train_steps=300 \
                        --seed="0" \
                        $text_encoder_option
                    
                    echo "Finished training for current configuration"
                    echo "----------------------------------------"
                done
            else
                echo "Running training for configuration:"
                echo "Dataset: $dataset"
                echo "Data root: $data_root"
                echo "Text encoder: ${text_encoder_option:-disabled}"
                
                CUDA_VISIBLE_DEVICES=0 accelerate launch train_text_to_image_lora.py \
                    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
                    --instance_data_dir="${data_root}/${dataset}" \
                    --output_dir="./checkpoint/${dataset}_${output_suffix}" \
                    --instance_prompt="$instance_prompt" \
                    --resolution=512 \
                    --train_batch_size=2 \
                    --gradient_accumulation_steps=1 \
                    --checkpointing_steps=1000 \
                    --learning_rate=5e-5 \
                    --lr_scheduler="constant" \
                    --lr_warmup_steps=0 \
                    --max_train_steps=500 \
                    --seed="0" \
                    $text_encoder_option
                
                echo "Finished training for current configuration"
                echo "----------------------------------------"
            fi
        done
    done
done