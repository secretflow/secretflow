# Data Preparation    Input: ./data
bash poison_gen.sh            # Output: ./advdm_data  ./antidb_data
bash diffpure.sh              # Output: ./purified_diffpure  
bash gridpure.sh              # Output: ./purified_gridpure

# LoRA fine-tuning   Output: ./checkpoint
bash train_text_to_image_lora_poison_clean.sh
bash train_text_to_image_lora_pure.sh

Dreambooth fine-tuning   Output: ./checkpoint_dreambooth
bash train_dreambooth_poison_clean.sh
bash train_dreambooth_pure.sh

LoRA generating   Output: ./output
bash generate.sh

# Dreambooth generating   Output: ./output_dreambooth
bash generate_dreambooth.sh

# Evaluating all fine-tuning configuration  fid/clip_iqa
bash evaluate_gen.sh

# Evaluating purified image psnr/ssim
bash evaluate_pure.sh