# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyright: reportMissingModuleSource=false

import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

def psnr(x, y, img_space='vqgan'):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))

def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)
