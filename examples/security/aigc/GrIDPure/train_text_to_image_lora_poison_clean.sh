#!/bin/bash

# Dataset configuration
datasets=("celebahq-caption" "wikiart")
# datasets=("wikiart" "celebahq-caption")
# Data directory configuration  
data_roots=("./data" "./advdm_data" "./antidb_data")
# Text encoder configuration
text_encoder_options=("" "--train_text_encoder")

# Corresponding checkpoint suffixes
checkpoint_suffixes=("clean" "advdm" "antidb")

for dataset in "${datasets[@]}"; do
   # Set corresponding prompt
   if [ "$dataset" = "wikiart" ]; then
       instance_prompt="a painting in the style of Pablo Picasso"
   else
       instance_prompt="a photography of a woman with long blonde hair and blue eyes"
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
               --train_batch_size=1 \
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
       done
   done
done