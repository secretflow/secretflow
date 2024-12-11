import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPImageQualityAssessment
from pathlib import Path
import os

class ImageEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048).to(device)
        self.clip_iqa = CLIPImageQualityAssessment().to(self.device)
        self.image_preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_images(self, image_dir, max_images=None):
        images = []
        image_paths = list(Path(image_dir).glob('*.png')) + list(Path(image_dir).glob('*.jpg'))
        if max_images:
            image_paths = image_paths[:max_images]
        
        for img_path in tqdm(image_paths, desc=f"Loading images from {image_dir}"):
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        return images

    def calculate_fid(self, real_images, generated_images):
        self.fid.reset()
        
        for img in tqdm(real_images, desc="Processing real images for FID"):
            img_tensor = self.image_preprocess(img)
            img_tensor = (img_tensor * 255).to(torch.uint8)
            img_tensor = img_tensor.unsqueeze(0)
            self.fid.update(img_tensor.to(self.device), real=True)
        
        for img in tqdm(generated_images, desc="Processing generated images for FID"):
            img_tensor = self.image_preprocess(img)
            img_tensor = (img_tensor * 255).to(torch.uint8)
            img_tensor = img_tensor.unsqueeze(0)
            self.fid.update(img_tensor.to(self.device), real=False)
        
        return float(self.fid.compute())

    def calculate_clip_iqa(self, images):
        scores = []
        
        for img in tqdm(images, desc="Calculating CLIP IQA scores"):
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.Resampling.LANCZOS)

            img_tensor = self.image_preprocess(img)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                score = self.clip_iqa(img_tensor)
                scores.append(score.item())
        
        return np.mean(scores)

def main():
    parser = argparse.ArgumentParser(description='Calculate image metrics')
    parser.add_argument('--real_dir', type=str, required=True, help='Directory containing real images')
    parser.add_argument('--generated_dir', type=str, required=True, help='Directory containing generated images')
    parser.add_argument('--output_file', type=str, default='metrics_results.txt', help='Output file for metrics')
    
    args = parser.parse_args()
    
    evaluator = ImageEvaluator()

    real_images = evaluator.load_images(args.real_dir)
    generated_images = evaluator.load_images(args.generated_dir)
    
    print(f"Loaded {len(real_images)} real images and {len(generated_images)} generated images")
    
    metrics = {}
    
    print("\nCalculating FID score...")
    metrics['fid'] = evaluator.calculate_fid(real_images, generated_images)
    
    print("\nCalculating CLIP IQA score...")
    metrics['clip_iqa'] = evaluator.calculate_clip_iqa(generated_images)
    
    with open(args.output_file, 'w') as f:
        for metric, value in metrics.items():
            print(f"\n{metric}: {value}")
            f.write(f"{metric}: {value}\n")

if __name__ == "__main__":
    main()