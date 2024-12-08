import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable
from copy import deepcopy
import numpy as np

# Importing models and loss provider
sys.path.append('src')
from src.ldm.models.autoencoder import AutoencoderKL
from src.ldm.models.diffusion.ddpm import LatentDiffusion
from src.loss.loss_provider import LossProvider
# Importing utility functions
import utils
import utils_img
import utils_model

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser()

    # Data parameters
    group = parser.add_argument_group('Data parameters')
    group.add_argument("--train_dir", type=str, required=True, help="Path to the training data directory")
    group.add_argument("--val_dir", type=str, required=True, help="Path to the validation data directory")

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument("--ldm_config", type=str, default="configs/v2-inference.yaml", help="Path to the LDM config")
    group.add_argument("--ldm_ckpt", type=str, default="models/v2-1_512-ema-pruned.ckpt", help="Path to the LDM checkpoint")
    group.add_argument("--msg_decoder_path", type=str, default="models/dec_48b_whit.torchscript.pt", help="Path to the watermarking model's hidden decoder")
    group.add_argument("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    group.add_argument("--redundancy", type=int, default=1, help="Redundancy of watermark repetition")
    group.add_argument("--decoder_depth", type=int, default=8, help="Depth of the watermarking decoder")
    group.add_argument("--decoder_channels", type=int, default=64, help="Number of channels in the watermarking decoder")

    # Training parameters
    group = parser.add_argument_group('Training parameters')
    group.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    group.add_argument("--img_size", type=int, default=256, help="Resize images to this size")
    group.add_argument("--loss_i", type=str, default="watson-vgg", help="Type of image loss function")
    group.add_argument("--loss_w", type=str, default="bce", help="Type of watermark loss function")
    group.add_argument("--lambda_i", type=float, default=0.2, help="Weight of image loss in total loss")
    group.add_argument("--lambda_w", type=float, default=1.0, help="Weight of watermark loss in total loss")
    group.add_argument("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate")
    group.add_argument("--steps", type=int, default=100, help="Number of training steps")
    group.add_argument("--warmup_steps", type=int, default=20, help="Number of warmup steps for optimizer")

    # Logging and saving parameters
    group = parser.add_argument_group('Logging and saving parameters')
    group.add_argument("--log_freq", type=int, default=10, help="Logging frequency (steps)")
    group.add_argument("--save_img_freq", type=int, default=1000, help="Image saving frequency (steps)")

    # Experiment parameters
    group = parser.add_argument_group('Experiment parameters')
    group.add_argument("--num_keys", type=int, default=5, help="Number of fine-tuned checkpoints")
    group.add_argument("--output_dir", type=str, default="output/", help="Output directory for logs and images")
    group.add_argument("--seed", type=int, default=0, help="Random seed")
    group.add_argument("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def load_models(params):
    """Load the LDM model and watermark decoder."""
    print(f"Loading LDM model from {params.ldm_config} and checkpoint {params.ldm_ckpt}...")
    config = OmegaConf.load(params.ldm_config)
    ldm_ae = utils_model.load_model_from_config(config, params.ldm_ckpt).first_stage_model.to(device)
    ldm_ae.eval()

    print(f"Loading watermark decoder from {params.msg_decoder_path}...")
    msg_decoder = load_msg_decoder(params)
    return ldm_ae, msg_decoder


def load_msg_decoder(params):
    """Load and possibly whiten the watermarking model decoder."""
    if 'torchscript' in params.msg_decoder_path:
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
    else:
        msg_decoder = utils_model.get_hidden_decoder(
            num_bits=params.num_bits, 
            redundancy=params.redundancy, 
            num_blocks=params.decoder_depth, 
            channels=params.decoder_channels
        ).to(device)
        msg_decoder.load_state_dict(utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path), strict=False)
        msg_decoder = whiten_decoder(msg_decoder, params)
        msg_decoder.eval()
    return msg_decoder


def whiten_decoder(msg_decoder, params):
    """Whiten the decoder to improve robustness."""
    print("Whitening watermark decoder...")
    with torch.no_grad():
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        loader = utils.get_dataloader(params.train_dir, transform, batch_size=16)
        ys = [msg_decoder(x.to(device)) for x in loader]
        ys = torch.cat(ys, dim=0)
        
        mean = ys.mean(dim=0, keepdim=True)
        ys_centered = ys - mean
        cov = ys_centered.T @ ys_centered
        e, v = torch.linalg.eigh(cov)
        L = torch.diag(1.0 / torch.pow(e, 0.5))
        weight = torch.mm(L, v.T)
        bias = -torch.mm(mean, weight.T).squeeze(0)

        # Apply whitening transformation
        linear = nn.Linear(ys.shape[1], ys.shape[1], bias=True)
        linear.weight.data = np.sqrt(ys.shape[1]) * weight
        linear.bias.data = np.sqrt(ys.shape[1]) * bias
        msg_decoder = nn.Sequential(msg_decoder, linear.to(device))

        # Save the whitened model
        torch.jit.save(torch.jit.script(msg_decoder), params.msg_decoder_path.replace(".pth", "_whit.pth"))
    return msg_decoder


def get_loss_functions(params):
    """Create and return the loss functions for watermark and image."""
    # Watermark loss
    if params.loss_w == 'mse':
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded * temp - (2 * keys - 1)) ** 2)
    elif params.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')
    else:
        raise NotImplementedError(f"Watermark loss function {params.loss_w} not implemented.")

    # Image loss
    if params.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)
    elif params.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif params.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif params.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    else:
        raise NotImplementedError(f"Image loss function {params.loss_i} not implemented.")

    return loss_w, loss_i

def create_key(nbit, device):
    """Generate a random binary key."""
    print(f'\n>>> Creating key with {nbit} bits...')
    key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
    key_str = "".join([str(int(ii)) for ii in key.tolist()[0]])
    print(f'Key: {key_str}')
    return key, key_str

def prepare_ldm_decoder(ldm_ae, device):
    """Prepare a copy of the LDM model decoder for fine-tuning."""
    print(f'>>> Preparing LDM decoder for fine-tuning...')
    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.encoder = nn.Identity()
    ldm_decoder.quant_conv = nn.Identity()
    ldm_decoder.to(device)
    for param in ldm_decoder.parameters():
        param.requires_grad = True
    return ldm_decoder


def save_checkpoint(output_dir, save_dict, log_stats, key_str, ii_key):
    """Save model checkpoints and logs."""
    torch.save(save_dict, os.path.join(output_dir, f"checkpoint_{ii_key:03d}.pth"))
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")
    with open(os.path.join(output_dir, "keys.txt"), "a") as f:
        f.write(f"{os.path.join(output_dir, f'checkpoint_{ii_key:03d}.pth')}\t{key_str}\n")

def encode_and_decode(imgs, ldm_ae, ldm_decoder):
    """Encode and decode images using the autoencoder and the finetuned decoder."""
    imgs_z = ldm_ae.encode(imgs).mode()  # b c h w -> b z h/f w/f
    imgs_d0 = ldm_ae.decode(imgs_z)  # b z h/f w/f -> b c h w
    imgs_w = ldm_decoder.decode(imgs_z)  # b z h/f w/f -> b c h w
    return imgs_d0, imgs_w


def compute_accuracy(decoded, keys):
    """Compute bit and word accuracy."""
    diff = (~torch.logical_xor(decoded > 0, keys > 0))
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
    word_accs = (bit_accs == 1)
    return bit_accs, word_accs

def log_stat(metric_logger, optimizer, bit_accs, word_accs, loss, lossw, lossi, imgs_w, imgs_d0, ii):
    """Log statistics to MetricLogger and print."""
    log_stats = {
        "iteration": ii,
        "loss": loss.item(),
        "loss_w": lossw.item(),
        "loss_i": lossi.item(),
        "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
        "bit_acc_avg": torch.mean(bit_accs).item(),
        "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
        "lr": optimizer.param_groups[0]["lr"],
    }
    for name, value in log_stats.items():
        metric_logger.update(**{name: value})
    return log_stats

def save_images(imgs, imgs_d0, imgs_w, ii, params):
    """Save the images to disk."""
    save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1), 
               os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
    save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1), 
               os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
    save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1), 
               os.path.join(params.imgs_dir, f'{ii:03}_train_w.png'), nrow=8)

def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable,
          ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet: nn.Module,
          key: torch.Tensor, params: argparse.Namespace):
    """Train the model."""
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        keys = key.repeat(imgs.shape[0], 1)

        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)

        # Encode images, decode latents with original and finetuned decoder
        imgs_d0, imgs_w = encode_and_decode(imgs, ldm_ae, ldm_decoder)

        # extract watermark
        decoded = msg_decoder(vqgan_to_imnet(imgs_w)) # b c h w -> b k

        # compute loss
        lossw = loss_w(decoded, keys)
        lossi = loss_i(imgs_w, imgs_d0)
        loss = params.lambda_w * lossw + params.lambda_i * lossi

        # Update optimizer
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute accuracy
        bit_accs, word_accs = compute_accuracy(decoded, keys)

        # Log statistics
        log_stats = log_stat(metric_logger, optimizer, bit_accs, word_accs, loss, lossw, lossi, imgs_w, imgs_d0, ii)
        
        if ii % params.log_freq == 0:
            print(json.dumps(log_stats))

        # Save images during training
        if ii % params.save_img_freq == 0:
            save_images(imgs, imgs_d0, imgs_w, ii, params)

    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module,
        vqgan_to_imnet: nn.Module, key: torch.Tensor, params: argparse.Namespace):
    """Evaluate the model."""
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    
    attacks = {
        'none': lambda x: x,
        'crop_01': lambda x: utils_img.center_crop(x, 0.1),
        'crop_05': lambda x: utils_img.center_crop(x, 0.5),
        'rot_25': lambda x: utils_img.rotate(x, 25),
        'rot_90': lambda x: utils_img.rotate(x, 90),
        'resize_03': lambda x: utils_img.resize(x, 0.3),
        'resize_07': lambda x: utils_img.resize(x, 0.7),
        'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
        'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
        'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
        'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
    }
    
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        imgs_d0, imgs_w = encode_and_decode(imgs, ldm_ae, ldm_decoder)

        keys = key.repeat(imgs.shape[0], 1)
        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
        }

        # Log attack results
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w))
            decoded = msg_decoder(imgs_aug)  # b c h w -> b k
            bit_accs, word_accs = compute_accuracy(decoded, keys)
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
            for name, value in log_stats.items():
                metric_logger.update(**{name: value})
        
        # Save images
        if ii % params.save_img_freq == 0:
            save_images(imgs, imgs_d0, imgs_w, ii, params)
    
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def main(params):
    """Main training function."""
    # Set random seeds for reproducibility
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

    # Create necessary directories
    os.makedirs(params.output_dir, exist_ok=True)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    os.makedirs(imgs_dir, exist_ok=True)
    params.imgs_dir = imgs_dir

    # Load models
    ldm_ae, msg_decoder = load_models(params)
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]
    # Freeze LDM and hidden decoder
    for param in [*msg_decoder.parameters(), *ldm_ae.parameters()]:
        param.requires_grad = False

    # Loads the data
    print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    # Create loss functions
    loss_w, loss_i = get_loss_functions(params)



    
    for ii_key in range(params.num_keys):
        # Create a random key for each iteration
        key, key_str = create_key(nbit=nbit, device=device)

        # Prepare the LDM decoder for fine-tuning
        ldm_decoder = prepare_ldm_decoder(ldm_ae, device)

        # Create optimizer for the LDM decoder
        optim_params = utils.parse_params(params.optimizer)
        optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

        # # Training loop
        print(f'>>> Training...')
        train_stats = train(train_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        val_stats = val(val_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
        log_stats = {'key': key_str,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
            }
        save_dict = {
            'ldm_decoder': ldm_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }

        # Save the checkpoint and log statistics
        save_checkpoint(params.output_dir, save_dict, log_stats, key_str, ii_key)

        # Optionally print the statistics
        print(f"Training and validation complete for key {ii_key + 1}/{params.num_keys}.")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
