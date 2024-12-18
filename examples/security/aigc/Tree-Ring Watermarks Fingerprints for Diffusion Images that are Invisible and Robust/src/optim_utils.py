import torch
from torchvision import transforms
from datasets import load_dataset

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy
import pandas as pd
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)
        
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# for one prompt to multiple images
def measure_similarity(images, prompt, args, device):
    model, preprocess, tokenizer = load_reference_model(args,device)
    with torch.no_grad():
        img_batch = [preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = pd.read_parquet('prompts/eval.parquet')
        prompt_key = ''

    return dataset, prompt_key

def setup_pipeline(args, device):
    """Set up the stable diffusion pipeline."""
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipeline = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
    )
    return pipeline.to(device)


def load_reference_model(args, device):
    """Load a reference model for similarity evaluation, if specified."""
    if args.reference_model:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.reference_model, 
            pretrained=args.reference_model_pretrain, 
            device=device
        )
        tokenizer = open_clip.get_tokenizer(args.reference_model)
        return model, preprocess, tokenizer
    return None, None, None


def generate_circle_mask(size=64, radius=10, x0=0, x_offset=0, y0=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= radius**2


def create_watermark_mask(watermark_pattern, args, device):
    """
    Create a watermarking mask based on the specified shape and parameters.
    """
    mask = torch.zeros(watermark_pattern.shape, dtype=torch.bool).to(device)

    if args.watermark_shape == 'circle':
        # Create a circular mask
        circle = generate_circle_mask(watermark_pattern.shape[-1], radius=args.watermark_radius)
        torch_circle = torch.tensor(circle).to(device)

        if args.watermark_channel == -1:
            # Apply the mask to all channels
            mask[:, :] = torch_circle
        else:
            # Apply the mask to a specific channel
            mask[:, args.watermark_channel] = torch_circle

    elif args.watermark_shape == 'square':
        # Create a square mask
        center = watermark_pattern.shape[-1] // 2
        radius = args.watermark_radius
        if args.watermark_channel == -1:
            # Apply the square mask to all channels
            mask[:, :, center - radius:center + radius, center - radius:center + radius] = True
        else:
            # Apply the square mask to a specific channel
            mask[:, args.watermark_channel, center - radius:center + radius, center - radius:center + radius] = True

    elif args.watermark_shape == 'none':
        # No mask is applied
        pass

    else:
        raise ValueError(f"Unsupported mask shape: {args.watermark_shape}")

    return mask


def generate_watermark_pattern(pipe, args, device, latent_shape=None):
    """
    Generate a ground-truth watermark pattern based on specified configurations.
    """
    set_random_seed(args.watermark_seed)

    # Initialize the base tensor
    if latent_shape is not None:
        latent_init = torch.randn(*latent_shape, device=device)
    else:
        latent_init = pipe.get_random_latents()

    if 'seed_ring' in args.watermark_pattern:
        watermark_pattern = _apply_ring_pattern(latent_init, args, device, mode='seed_ring')
    elif 'seed_zeros' in args.watermark_pattern:
        watermark_pattern = latent_init * 0
    elif 'seed_rand' in args.watermark_pattern:
        watermark_pattern = latent_init
    elif 'rand' in args.watermark_pattern:
        watermark_pattern = _apply_frequency_domain_pattern(latent_init, mode='random')
    elif 'zeros' in args.watermark_pattern:
        watermark_pattern = _apply_frequency_domain_pattern(latent_init, mode='zeros')
    elif 'const' in args.watermark_pattern:
        watermark_pattern = _apply_frequency_domain_pattern(latent_init, mode='constant', constant_value=args.watermark_constant)
    elif 'ring' in args.watermark_pattern:
        watermark_pattern = _apply_ring_pattern(latent_init, args, device, mode='ring')
    else:
        raise ValueError(f"Unsupported watermark pattern: {args.watermark_pattern}")

    return watermark_pattern

def _apply_ring_pattern(latent_tensor, args, device, mode='ring'):
    """
    Apply a seed ring pattern to the latent tensor.
    """
    if mode == 'seed_ring':
        pattern = latent_tensor.clone()
    elif mode == 'ring':
        pattern = torch.fft.fftshift(torch.fft.fft2(latent_tensor), dim=(-1, -2))

    temp_pattern = copy.deepcopy(pattern)

    for radius in range(args.watermark_radius, 0, -1):
        circular_mask = generate_circle_mask(latent_tensor.shape[-1], radius=radius)
        torch_mask = torch.tensor(circular_mask).to(device)

        for channel in range(pattern.shape[1]):
            pattern[:, channel, torch_mask] = temp_pattern[0, channel, 0, radius].item()

    return pattern

def _apply_frequency_domain_pattern(latent_tensor, mode='random', constant_value=0):
    """
    Apply a frequency domain pattern to the latent tensor.
    """
    freq_domain = torch.fft.fftshift(torch.fft.fft2(latent_tensor), dim=(-1, -2))

    if mode == 'random':
        return freq_domain.clone()
    elif mode == 'zeros':
        return freq_domain * 0
    elif mode == 'constant':
        return freq_domain * 0 + constant_value
    else:
        raise ValueError(f"Unsupported frequency domain mode: {mode}")
    
def compute_watermark_metrics(pipe, distorted_no_watermark, distorted_with_watermark, 
                              prompt_embedding, watermark_pattern, watermark_mask, args, device):
    """
    Compute the watermark detection metrics.
    """

    # Convert images to latents
    latents_no_watermark = pipe.get_image_latents(transform_img(distorted_no_watermark).unsqueeze(0).to(prompt_embedding.dtype).to(device), sample=False)
    latents_with_watermark = pipe.get_image_latents(transform_img(distorted_with_watermark).unsqueeze(0).to(prompt_embedding.dtype).to(device), sample=False)

    # Reverse diffusion to get reconstructed latents from the images
    reversed_latents_no_watermark = pipe.forward_diffusion(
        latents=latents_no_watermark,
        text_embeddings=prompt_embedding,
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
    )

    reversed_latents_with_watermark = pipe.forward_diffusion(
        latents=latents_with_watermark,
        text_embeddings=prompt_embedding,
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
    )

    no_watermark_metric, with_watermark_metric = eval_watermark(reversed_latents_no_watermark, reversed_latents_with_watermark, watermark_pattern, watermark_mask, args)

    return no_watermark_metric, with_watermark_metric


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.watermark_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.watermark_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'watermark_injection: {args.watermark_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermark_pattern, watermarking_mask, args):
    if 'complex' in args.watermark_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = watermark_pattern
    elif 'seed' in args.watermark_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = watermark_pattern
    else:
        NotImplementedError(f'watermark_measurement: {args.watermark_measurement}')

    if 'l1' in args.watermark_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'watermark_measurement: {args.watermark_measurement}')

    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w
