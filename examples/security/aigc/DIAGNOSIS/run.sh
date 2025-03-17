#!/bin/bash

# Function to run the complete pipeline for a given configuration
run_pipeline() {
    local model_name=$1      # "sd1.5" or "sd2"
    local dataset=$2         # "celeba" or "mscoco"
    local protection=$3      # "protected" or "unprotected"
    local condition=$4       # "conditional" or "unconditional"
    
    # Set model path based on version
    local model_path
    local model_prefix
    if [ "$model_name" = "sd1.5" ]; then
        model_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
        model_prefix="sdv1-5"
    else
        model_path="stabilityai/stable-diffusion-2"
        model_prefix="sdv2"
    fi
    
    # Set dataset full name
    local dataset_full
    if [ "$dataset" = "celeba" ]; then
        dataset_full="celeba_with_llava_captions"
    else
        dataset_full="wds_mscoco_captions2017"
    fi
    
    # Set parameters based on protection and condition
    local p_value="0.0"
    local wanet_suffix=""
    local s_value="1.0"
    local trigger_flag=""
    
    if [ "$protection" = "protected" ]; then
        if [ "$condition" = "conditional" ]; then
            p_value="0.2"
            wanet_suffix="_wanet"
            trigger_flag="--trigger_conditioned"
        else
            p_value="1.0"
            wanet_suffix="_wanet_unconditional"
            s_value="2.0"
        fi
    else
        wanet_suffix="_none"
    fi
    
    # Construct paths
    local data_suffix="_removeeval"
    [ "$protection" = "protected" ] && data_suffix="_s${s_value}_k128_removeeval"
    
    local train_data_dir="./coated_data/${dataset_full}_p${p_value}${wanet_suffix}${data_suffix}"
    
    local checkpoint_dir="./checkpoint/lora_${model_prefix}_${dataset}_p${p_value}${wanet_suffix}${data_suffix}"
    local output_dir="output/${model_prefix}_${dataset}_p${p_value}${wanet_suffix}${data_suffix}"
    
    echo "Running pipeline for: $model_name $dataset $protection $condition"
    echo "Data dir: $train_data_dir"
    echo "Checkpoint dir: $checkpoint_dir"
    echo "Output dir: $output_dir"
    
    # Training
    CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
        --pretrained_model_name_or_path="$model_path" \
        --train_data_dir="$train_data_dir" \
        --caption_column="additional_feature" \
        --resolution=512 \
        --random_flip \
        --train_batch_size=1 \
        --num_train_epochs=20 \
        --checkpointing_steps=10000 \
        --learning_rate=1e-04 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --seed=42 \
        --output_dir="$checkpoint_dir" \
        --validation_prompt=None
        
    # Generation
    CUDA_VISIBLE_DEVICES=0 python generate.py \
        --model_type "$model_path" \
        --model_checkpoint "$checkpoint_dir" \
        --save_path "$output_dir" \
        --dataset_name "$dataset"
        
    # Binary Classification
    CUDA_VISIBLE_DEVICES=0 python binary_classifier.py \
        --ori_dir "./coated_data/${dataset_full}_p0.0_none_removeeval/train" \
        --coated_dir "./coated_data/${dataset_full}_p1.0_wanet_unconditional_s2.0_k128_removeeval/train" \
        --generated_inspected_dir "$output_dir" \
        --dataset_method "${dataset}_wanet_2.0" \
        $trigger_flag
        
    echo "Completed pipeline for: $model_name $dataset $protection $condition"
    echo "----------------------------------------"
}

# Run all combinations
for model in "sd1.5" "sd2"; do
    for dataset in "celeba" "mscoco"; do
        # Protected cases
        run_pipeline "$model" "$dataset" "protected" "unconditional"
        run_pipeline "$model" "$dataset" "protected" "conditional"
        
        # Unprotected case
        run_pipeline "$model" "$dataset" "unprotected" "unconditional"
    done
done

echo "All experiments completed!"