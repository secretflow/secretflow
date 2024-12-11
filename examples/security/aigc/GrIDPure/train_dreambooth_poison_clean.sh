#!/bin/bash

# Dataset configuration
datasets=("wikiart" "celebahq-caption")
# Data directory configuration
data_roots=("./data" "./advdm_data" "./antidb_data")
# train_text_encoder configuration
text_encoder_options=("" "--train_text_encoder")
# checkpoint suffix
checkpoint_suffixes=("clean" "advdm" "antidb")

for dataset in "${datasets[@]}"; do
    # Set corresponding prompt
    if [ "$dataset" = "wikiart" ]; then
        instance_prompt="a painting in the style of PCS"
        class_prompt="a painting"
    else
        instance_prompt="a photography of a sks person"
        class_prompt="a photography of a person"
    fi
    
    # Iterate through data directories
    for i in "${!data_roots[@]}"; do
        data_root="${data_roots[$i]}"
        checkpoint_suffix="${checkpoint_suffixes[$i]}"
        
        # Iterate through text_encoder options
        for text_encoder_option in "${text_encoder_options[@]}"; do
            # Build output directory name
            if [ -n "$text_encoder_option" ]; then
                output_suffix="${checkpoint_suffix}_with_text_encoder"
            else
                output_suffix="${checkpoint_suffix}"
            fi
            
            # Build checkpoint path
            checkpoint_dir="./checkpoint_dreambooth/${dataset}_${output_suffix}"
            
            # Check if checkpoint already exists
            if [ -d "$checkpoint_dir" ] && [ -f "${checkpoint_dir}/scheduler/scheduler_config.json" ]; then
                echo "Checkpoint already exists at $checkpoint_dir. Skipping..."
                continue
            fi
            
            # Create class images directory
            class_dir="./class_images/${dataset}"
            mkdir -p "$class_dir"
            
            echo "Running training for configuration:"
            echo "Dataset: $dataset"
            echo "Data root: $data_root"
            echo "Text encoder: ${text_encoder_option:-disabled}"
            
            CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth.py \
                --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
                --instance_data_dir="${data_root}/${dataset}" \
                --class_data_dir="$class_dir" \
                --output_dir="$checkpoint_dir" \
                --instance_prompt="$instance_prompt" \
                --class_prompt="$class_prompt" \
                --with_prior_preservation \
                --prior_loss_weight=1.0 \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=1 \
                --learning_rate=5e-6 \
                --lr_scheduler="constant" \
                --lr_warmup_steps=0 \
                --num_class_images=200 \
                --max_train_steps=500 \
                --seed="0" \
                --use_8bit_adam \
                $text_encoder_option
                
            echo "Finished training for current configuration"
            echo "----------------------------------------"
        done
    done
done