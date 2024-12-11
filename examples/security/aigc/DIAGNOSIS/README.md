# DIAGNOSIS Algorithm Implementation
This project reproduces the ICLR 2024 paper ["DIAGNOSIS: Detecting Unauthorized Data Usages in Text-to-image Diffusion Models"](https://openreview.net/pdf?id=f8S3aLm0Vp).

## Project Structure
```
├── bc_checkpoint/          # Binary classifier model parameters for watermark detection
├── checkpoint/            # LoRA module parameters from model fine-tuning
├── coated_data/          # Watermarked data storage
├── data/                 # Original dataset directory
├── dataset/              # Dataset processing code
├── output/               # Generated images using fine-tuned LoRA modules
Configuration files:
├── .gitignore          
├── requirements.txt     
├── README.md           
Python files:
├── evaluate.py           # Results aggregation
├── binary_classifier.py  # Binary classifier implementation
├── coating.py           # Data watermarking
├── generate.py          # Image generation with fine-tuned models
└── train_text_to_image_lora.py  # LoRA fine-tuning
Shell scripts:
├── coating.sh           # Data watermarking workflow
├── run_ablation.sh      # Ablation studies
├── run_all.sh           # Run all experiments
└── run.sh               # Comparative experiments
```

## Environment Setup
1. Clone this repository
```bash
git clone [repository_url]
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation
1. Create dataset directories under the `data` folder
2. Download and process datasets:
```bash
# Process CelebA dataset
python dataset/load_dataset_celeba.py

# Process MSCOCO dataset
python dataset/load_dataset_coco.py
```

## Experimental Pipeline
### 1. Data Processing
- Process CELEBA and MSCOCO datasets
- Apply both unconditional and conditional watermarking
- Test different watermark strengths and coverage rates
```bash
bash coating.sh
```

### 2. Main Experiments
- Test different models (SD v1.5, SD v2)
- Evaluate on different datasets
- Compare various protection methods
```bash
bash run.sh
```
Execution steps:
1. Model training (using LoRA method)
2. Image generation
3. Detection using binary classifier
4. Calculate memorization strength and FID scores

### 3. Ablation Studies
This script performs various ablation studies to analyze different parameters:
- Test effects of watermark strength
- Evaluate different coverage rates
- Verify model performance under different configurations
```bash
bash run_ablation.sh
```

## Custom Examples
You can customize the training pipeline according to your specific needs by preparing your own data in the data/ directory.

### 1. Add Watermarks
```bash
# Unconditional watermark
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 2 \
    --remove_eval --number_to_coat 2000

# Trigger-conditioned watermark
python coating.py --p 0.2 --target_type wanet --wanet_s 1 \
    --remove_eval --number_to_coat 2000
```

### 2. Model Fine-tuning
```bash
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
```

### 3. Image Generation
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_type $MODEL_NAME \
    --model_checkpoint $CHECKPOINT_DIR \
    --save_path $OUTPUT_DIR \
    --dataset_name $DATASET_NAME
```

### 4. Detect Unauthorized Usage
```bash
CUDA_VISIBLE_DEVICES=0 python binary_classifier.py \
    --ori_dir "./coated_data/${dataset_full}_p0.0_none_removeeval/train" \
    --coated_dir "./coated_data/${dataset_full}_p1.0_wanet_unconditional_s2.0_k128_removeeval/train" \
    --generated_inspected_dir "$output_dir" \
    --dataset_method "${dataset}_wanet_2.0" \
    $trigger_flag
```

## Detailed Code Flow
For complete implementation details, refer to our scripts:
- `coating.sh`: Data perturbation
- `run_ablation.sh`: Ablation experiments
- `run.sh`: Main experiments