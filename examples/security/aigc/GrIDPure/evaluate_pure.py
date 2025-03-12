import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_images(image_dir):
    images = []
    image_paths = sorted(list(Path(image_dir).glob('*.png')) + list(Path(image_dir).glob('*.jpg')))
    
    for img_path in tqdm(image_paths, desc=f"Loading images from {image_dir}"):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return images

def calculate_metrics(real_images, purified_images):
    if len(real_images) != len(purified_images):
        print(f"Warning: Number of images different. Real: {len(real_images)}, Purified: {len(purified_images)}")
    
    num_images = min(len(real_images), len(purified_images))
    psnr_scores = []
    ssim_scores = []
    
    for i in tqdm(range(num_images), desc="Calculating metrics"):
        real_img = real_images[i]
        pure_img = purified_images[i]
        
        if pure_img.size != real_img.size:
            pure_img = pure_img.resize(real_img.size, Image.Resampling.LANCZOS)
        
        real_np = np.array(real_img)
        pure_np = np.array(pure_img)
        
        psnr_scores.append(psnr(real_np, pure_np, data_range=255))
        ssim_scores.append(ssim(real_np, pure_np, channel_axis=2, data_range=255))
    
    return np.mean(psnr_scores), np.mean(ssim_scores)

def main():
    parser = argparse.ArgumentParser(description='Evaluate purification metrics')
    parser.add_argument('--real_dir', type=str, required=True, help='Directory containing real images')
    parser.add_argument('--purified_dir', type=str, required=True, help='Directory containing purified images')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for metrics')
    
    args = parser.parse_args()
    
    real_images = load_images(args.real_dir)
    purified_images = load_images(args.purified_dir)
    
    psnr_value, ssim_value = calculate_metrics(real_images, purified_images)
    
    with open(args.output_file, 'w') as f:
        f.write(f"psnr: {psnr_value}\n")
        f.write(f"ssim: {ssim_value}\n")
    
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")

if __name__ == "__main__":
    main()