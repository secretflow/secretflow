#!/bin/bash

# Common paths
BASE_DIR="./data"
SD_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"

# CelebA-HQ paths
CELEBA_DATA="${BASE_DIR}/celebahq-caption"
CELEBA_CLASS_DATA="./antidb_data/celebahq-caption_class_data"
CELEBA_ADVDM_OUT="advdm_data/celebahq-caption"
CELEBA_ANTIDB_OUT="./antidb_data/celebahq-caption"

# WikiArt paths
WIKIART_DATA="${BASE_DIR}/wikiart"
WIKIART_CLASS_DATA="./antidb_data/wikiart_class_data"
WIKIART_ADVDM_OUT="advdm_data/wikiart"
WIKIART_ANTIDB_OUT="./antidb_data/wikiart"

# Run AdvDM for CelebA-HQ
echo "Running AdvDM for CelebA-HQ..."
CUDA_VISIBLE_DEVICES=0 python poison_adv.py \
  --pretrained_model_name_or_path="${SD_MODEL}" \
  --instance_data_dir="${CELEBA_DATA}" \
  --output_dir="${CELEBA_ADVDM_OUT}" \
  --poison_scale=8 \
  --poison_step_num=100 \
  --resolution=512 \
  --instance_prompt="a photography of a woman with long blonde hair and blue eyes" \
  --train_batch_size=1 \
  --mixed_precision="fp16"

# Run Anti-DreamBooth for CelebA-HQ
echo "Running Anti-DreamBooth for CelebA-HQ..."
CUDA_VISIBLE_DEVICES=0 accelerate launch poison_anti_db.py \
  --pretrained_model_name_or_path="${SD_MODEL}" \
  --instance_data_dir_for_train="${CELEBA_DATA}" \
  --instance_data_dir_for_adversarial="${CELEBA_DATA}" \
  --instance_prompt="a photography of a woman with long blonde hair and blue eyes" \
  --class_data_dir="${CELEBA_CLASS_DATA}" \
  --num_class_images=50 \
  --class_prompt="a photo of a woman" \
  --output_dir="${CELEBA_ANTIDB_OUT}" \
  --center_crop \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=2 \
  --max_train_steps=10 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=10 \
  --learning_rate=5e-7 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2

# Run AdvDM for WikiArt
echo "Running AdvDM for WikiArt..."
CUDA_VISIBLE_DEVICES=0 python poison_adv.py \
  --pretrained_model_name_or_path="${SD_MODEL}" \
  --instance_data_dir="${WIKIART_DATA}" \
  --output_dir="${WIKIART_ADVDM_OUT}" \
  --poison_scale=8 \
  --poison_step_num=100 \
  --resolution=512 \
  --instance_prompt="a painting in the style of Pablo Picasso" \
  --train_batch_size=1 \
  --mixed_precision="fp16"

# Run Anti-DreamBooth for WikiArt
echo "Running Anti-DreamBooth for WikiArt..."
CUDA_VISIBLE_DEVICES=0 accelerate launch poison_anti_db.py \
  --pretrained_model_name_or_path="${SD_MODEL}" \
  --instance_data_dir_for_train="${WIKIART_DATA}" \
  --instance_data_dir_for_adversarial="${WIKIART_DATA}" \
  --instance_prompt="a painting in the style of Pablo Picasso" \
  --class_data_dir="${WIKIART_CLASS_DATA}" \
  --num_class_images=50 \
  --class_prompt="a painting" \
  --output_dir="${WIKIART_ANTIDB_OUT}" \
  --center_crop \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=2 \
  --max_train_steps=10 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=10 \
  --learning_rate=5e-7 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2