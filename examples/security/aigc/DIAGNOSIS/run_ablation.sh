#!/bin/bash
run_pipeline() {
    local dataset=$1
    local p=$2
    local s=$3
    local uncond=$4
    
    # Construct directory names
    local uncond_suffix=""
    if [ "$uncond" = true ]; then
        uncond_suffix="_unconditional"
    fi
    
    local data_dir="./coated_data/${dataset}_p${p}_wanet${uncond_suffix}_s${s}_k128_removeeval"
    local checkpoint_dir="./checkpoint/lora_sdv1-5_${dataset}_p${p}_wanet${uncond_suffix}_s${s}_k128"
    local output_dir="output/sdv1-5_${dataset}_p${p}_wanet${uncond_suffix}_s${s}_k128"
    
    # Convert dataset name to shorter version for classifier
    local dataset_short="celeba"
    if [[ $dataset == "wds_mscoco_captions2017" ]]; then
        dataset_short="mscoco"
    fi
    
    echo "Running pipeline for ${dataset} p=${p} s=${s} uncond=${uncond}"
    
    # Training
    CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
        --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
        --train_data_dir="${data_dir}" \
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
        --output_dir="${checkpoint_dir}" \
        --validation_prompt=None
    
    # Generation
    CUDA_VISIBLE_DEVICES=0 python generate.py \
        --model_type "stable-diffusion-v1-5/stable-diffusion-v1-5" \
        --model_checkpoint "${checkpoint_dir}" \
        --save_path "${output_dir}" \
        --dataset_name "${dataset_short}"
    
    # Binary Classification
    CUDA_VISIBLE_DEVICES=0 python binary_classifier.py \
        --ori_dir "./coated_data/${dataset}_p0.0_none_removeeval/train" \
        --coated_dir "${data_dir}/train" \
        --generated_inspected_dir "${output_dir}" \
        --dataset_method "${dataset_short}_wanet_${s}"
}

# Different watermarking strengths
echo "Running experiments with different watermarking strengths..."
for dataset in "wds_mscoco_captions2017"; do  # celeba_with_llava_captions
    for s in 1.0 2.0 3.0 4.0; do
        run_pipeline "$dataset" "1.0" "$s" true
    done
done

# Different coating rates
echo "Running experiments with different coating rates..."
for dataset in "wds_mscoco_captions2017"; do  # celeba_with_llava_captions
    for p in 0.02 0.05 0.1 0.2 0.5; do
        # With unconditional
        run_pipeline "$dataset" "$p" "2.0" true
        run_pipeline "$dataset" "$p" "1.0" false
    done
done

echo "All experiments completed!"