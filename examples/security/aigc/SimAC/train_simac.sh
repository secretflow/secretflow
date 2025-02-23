export DATA_NAME="n000179"
export MODEL_NAME="models/stable-diffusion-2-1-base"
export INSTANCE_DIR="data/$DATA_NAME/set_A"
export CLEAN_DIR="data/$DATA_NAME/set_B"
export OUTPUT_DIR="outputs/$DATA_NAME"
export CLASS_DIR="class_images/class-person"

accelerate launch simac.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train=$CLEAN_DIR \
  --instance_data_dir_for_adversarial=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=200 \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --use_8bit_adam \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2 \
  --checkpointing_iterations=10 \
  --search_steps=50 \
  --search_delete=20
