import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import numpy as np
import torch
import sys

sys.path.append('src')
from optim_utils import *
from io_utils import *


def generate_images_with_and_without_watermark(pipe, prompt, seed, args, watermark_pattern, watermark_mask, device):
    """Generate images with and without watermarking."""
    set_random_seed(seed)
    latents_no_watermark = pipe.get_random_latents()

    # Generate image without watermark
    outputs_no_watermark = pipe(
        prompt,
        num_images_per_prompt=args.num_images,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.image_size,
        width=args.image_size,
        latents=latents_no_watermark,
    )
    image_no_watermark = outputs_no_watermark.images[0]

    # Generate image with watermark
    latents_with_watermark = copy.deepcopy(latents_no_watermark)
    latents_with_watermark = inject_watermark(
        latents_with_watermark, 
        watermark_mask, 
        watermark_pattern, 
        args
    )
    outputs_with_watermark = pipe(
        prompt,
        num_images_per_prompt=args.num_images,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.image_size,
        width=args.image_size,
        latents=latents_with_watermark,
    )
    image_with_watermark = outputs_with_watermark.images[0]


    return image_no_watermark, image_with_watermark


def evaluate_watermarking(pipe, dataset, args, device):
    """Evaluate the watermarking process across a dataset."""
    unknown_prompt_embedding = pipe.get_text_embedding('')
    watermark_pattern = generate_watermark_pattern(pipe, args, device)
    watermark_mask = create_watermark_mask(watermark_pattern, args, device)

    evaluation_results = []
    clip_scores_no_watermark, clip_scores_with_watermark = [], []
    metrics_no_watermark, metrics_with_watermark = [], []

    for index in tqdm(range(args.start_index, args.end_index)):
        seed = index + args.generation_seed
        prompt = dataset.Prompt[index]

        # Generate images
        image_no_watermark, image_with_watermark = generate_images_with_and_without_watermark(
            pipe, prompt, seed, args,
            watermark_pattern, watermark_mask, device
        )

        # Apply distortions
        distorted_no_watermark, distorted_with_watermark = image_distortion(
            image_no_watermark, image_with_watermark, seed, args
        )

        # Evaluate metrics
        no_watermark_metric, with_watermark_metric = compute_watermark_metrics(
            pipe, distorted_no_watermark, distorted_with_watermark, 
            unknown_prompt_embedding, watermark_pattern, watermark_mask, args, device
        )
        

        metrics_no_watermark.append(-no_watermark_metric)
        metrics_with_watermark.append(-with_watermark_metric)

        # Placeholder for CLIP-based scores (extendable if reference model is provided)
        if args.reference_model is not None:
            sims = measure_similarity([image_no_watermark, image_with_watermark], prompt, args, device)
            no_watermark_similarity = sims[0].item()
            with_watermark_similarity = sims[1].item()
        else:
            no_watermark_similarity = 0
            with_watermark_similarity = 0
        
        clip_scores_no_watermark.append(no_watermark_similarity)
        clip_scores_with_watermark.append(with_watermark_similarity)
        
        evaluation_results.append({
            'no_watermark_metric': no_watermark_metric,
            'with_watermark_metric': with_watermark_metric,
            'no_watermark_similarity': no_watermark_similarity,
            'with_watermark_similarity': with_watermark_similarity
        })

    return evaluation_results, metrics_no_watermark, metrics_with_watermark, clip_scores_no_watermark, clip_scores_with_watermark


def log_evaluation_results(args, results, clip_scores_no_w, clip_scores_with_w, metrics_no_w, metrics_with_w):
    """Log evaluation results to console and optionally to W&B."""
    predictions = metrics_no_w + metrics_with_w
    true_labels = [0] * len(metrics_no_w) + [1] * len(metrics_with_w)

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(true_labels, predictions)
    auc_score = metrics.auc(false_positive_rate, true_positive_rate)
    accuracy = np.max(1 - (false_positive_rate + (1 - true_positive_rate)) / 2)
    tpr_at_1_percent_fpr = true_positive_rate[np.where(false_positive_rate < 0.01)[0][-1]]

    print(f'Mean CLIP Score (No Watermark): {mean(clip_scores_no_w)}')
    print(f'Mean CLIP Score (With Watermark): {mean(clip_scores_with_w)}')
    print(f'AUC: {auc_score}, Accuracy: {accuracy}, TPR@1%FPR: {tpr_at_1_percent_fpr}')

    if args.enable_tracking:
        wandb.log({
            'Mean CLIP Score (No Watermark)': mean(clip_scores_no_w),
            'Mean CLIP Score (With Watermark)': mean(clip_scores_with_w),
            'AUC': auc_score,
            'Accuracy': accuracy,
            'TPR@1%FPR': tpr_at_1_percent_fpr,
        })


def main(args):
    if args.enable_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = setup_pipeline(args, device)
    dataset, _ = get_dataset(args)

    results, metrics_no_w, metrics_with_w, clip_scores_no_w, clip_scores_with_w = evaluate_watermarking(pipe, dataset, args, device)
    log_evaluation_results(args, results, clip_scores_no_w, clip_scores_with_w, metrics_no_w, metrics_with_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Watermarking')
    
    # General arguments
    parser.add_argument('--run_name', default='test_run', help='Name of the experiment run')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts', help='Dataset for prompts')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for evaluation')
    parser.add_argument('--end_index', type=int, default=10, help='End index for evaluation')
    parser.add_argument('--image_size', type=int, default=512, help='Generated image size')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base', help='Stable diffusion model ID')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--enable_tracking', action='store_true', help='Enable W&B tracking')

    # Generation parameters
    parser.add_argument('--num_images', type=int, default=1, help='Number of images per prompt')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--generation_seed', type=int, default=0, help='Seed for generation randomness')

    # Watermarking parameters
    parser.add_argument('--watermark_radius', type=int, default=10, help='Radius for watermark pattern')
    parser.add_argument('--watermark_channel', default=0, type=int)
    parser.add_argument('--watermark_pattern', default='ring')
    parser.add_argument('--watermark_shape', default='circle')
    parser.add_argument('--watermark_seed', type=int, default=999999, help='Seed for watermark generation')
    parser.add_argument('--watermark_injection', default='complex')
    parser.add_argument('--watermark_measurement', default='l1_complex')

    # Image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()
    main(args)
