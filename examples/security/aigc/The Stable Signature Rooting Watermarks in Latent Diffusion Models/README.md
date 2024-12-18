# The Stable Signature: Rooting Watermarks in Latent Diffusion Models
## Overview
This repository provides code and instructions to fine-tune an LDM (Latent Diffusion Model) decoder, generate watermarked images, and evaluate their robustness and quality. The workflow includes using the COCO dataset for training and running scripts for generation and evaluation.

## Dataset Preparation
### Requirements
- Use the COCO dataset as the base.
- Ensure around 500 images are available for training, preferably of size **256x256** or larger.

## Environment Setup
### Requirements
- Python 3.10
- PyTorch 1.12.0
- CUDA 12.4
### Installation
Install dependencies:
```
pip install -r requirements.txt
```

## Fine-Tune the LDM Decoder
Use the script `finetune.py` to fine-tune the LDM decoder.

### Command:
```
python finetune.py --num_keys 1 \
    --ldm_config path/to/ldm/config.yaml \
    --ldm_ckpt path/to/ldm/ckpt.pth \
    --msg_decoder_path path/to/msg/decoder/ckpt.torchscript.pt \
    --train_dir path/to/train/dir \
    --val_dir path/to/val/dir
```
### Notes:
- **ldm_config**: YAML configuration file for the LDM. Available in the `configs` directory.
- **ldm_ckpt**: Pretrained LDM checkpoint. You can download the base LDM model checkpoint v2-1_768-ema-pruned.ckpt from the stabilityai/stable-diffusion-2-1 repository on Hugging Face and place the checkpoint in the `models` directory.
- **msg_decoder_path**: Path to the pre-trained decoder checkpoint. A pre-trained message decoder checkpoint is provided in the Path: `models/dec_48b_whit.torchscript.pt`.

## Generate Watermarked Images
Use the `generaterImage.py` script to generate watermarked images. The fine-tuned LDM decoder weights can be downloaded from [this link](https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth).

### Command:
```
python generaterImage.py \
    --ldm_config configs/v2-inference.yaml \
    --ldm_ckpt models/v2-1_512-ema-pruned.ckpt \
    --image_dir outputs/imgs
```
### Notes:
- Generated images will be saved in the `outputs/imgs` directory.

## Decode and Evaluate
Use the `evaluation.py` script to evaluate watermarked images' robustness and quality metrics.

### Decode Bits and Evaluate Robustness:
```
python evaluation.py --eval_imgs False --eval_bits True \
    --img_dir path/to/imgs_w \
    --key_str '111010110101000001010111010011010100010000100111'\
    --msg_decoder_path models/dec_48b_whit.torchscript.pt
```
- **Output**: A CSV file with bit accuracy for various attack scenarios applied to the watermarked images.
### Compare Image Quality:
```
python evaluation.py --eval_imgs True --eval_bits False \
    --img_dir path/to/imgs_w \
    --img_dir_nw path/to/imgs_nw
```
- **Output**: A CSV file with image quality metrics:
  - **PSNR**: Peak Signal-to-Noise Ratio.
  - **SSIM**: Structural Similarity Index.
  - **LPIPS**: Learned Perceptual Image Patch Similarity.
## File Structure
- `configs/`: Contains YAML configuration files for the LDM.
- `models/`: Includes pretrained checkpoints.
- `outputs/`: Directory to store generated images and evaluation results.

## Usage
1. Prepare the dataset and split it into training and validation directories.
2. Fine-tune the LDM decoder using the provided script. 
3. Generate watermarked images using the `generaterImage.py` script.
4. Evaluate the robustness and quality of the watermarked images using the `evaluation.py`
