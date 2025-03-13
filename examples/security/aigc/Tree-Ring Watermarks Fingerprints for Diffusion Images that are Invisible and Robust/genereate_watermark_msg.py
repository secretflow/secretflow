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
    bit_accs = []

    for index in tqdm(range(args.start_index, args.end_index)):
        seed = index + args.generation_seed
        prompt = dataset.Prompt[index]

        # Generate images
        image_no_watermark, image_with_watermark = generate_images_with_and_without_watermark(
            pipe, prompt, seed, args,
            watermark_pattern, watermark_mask, device
        )

        image_path = '/data/code/wyp/ant_tree_ring/samples'
        if args.r_degree is not None:
            image_path = os.path.join(image_path, 'r_degree_{}'.format(args.r_degree))
        elif args.jpeg_ratio is not None:   
            image_path = os.path.join(image_path, 'jpeg_ratio_{}'.format(args.jpeg_ratio))
        elif args.crop_scale is not None:   
            image_path = os.path.join(image_path, 'crop_scale_{}'.format(args.crop_scale))
        if args.msg_scaler is not None:
            image_path = os.path.join(image_path, 'msg_scaler_{}'.format(args.msg_scaler))
        image_path_watermark = os.path.join(image_path, 'watermark')
        image_path_no_watermark = os.path.join(image_path, 'no_watermark')
        if not os.path.exists(image_path_watermark):
            os.makedirs(image_path_watermark, exist_ok=True)
        if not os.path.exists(image_path_no_watermark):
            os.makedirs(image_path_no_watermark, exist_ok=True)
        image_no_watermark.save('{}/{}.png'.format(image_path_no_watermark, index))
        image_with_watermark.save('{}/{}.png'.format(image_path_watermark, index))


        # Apply distortions
        distorted_no_watermark, distorted_with_watermark = image_distortion(
            image_no_watermark, image_with_watermark, seed, args
        )

        # Evaluate metrics
        no_watermark_metric, with_watermark_metric, reversed_latents_with_watermark = compute_watermark_metrics(
            pipe, distorted_no_watermark, distorted_with_watermark, 
            unknown_prompt_embedding, watermark_pattern, watermark_mask, args, device
        )
        
        if args.msg is not None:

            
            true_msg = np.array(list(map(lambda x: 1 if x == "1" else 0, list(args.msg))))
            pred_msg = np.array(detect_msg(reversed_latents_with_watermark, args).cpu())
            print(f"true msg: {true_msg}; pred_msg: {pred_msg}")

            correct_bits_tmp = np.equal(true_msg, pred_msg).sum()

            bit_acc = correct_bits_tmp / len(true_msg)

            bit_accs.append(bit_acc)


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
            'with_watermark_similarity': with_watermark_similarity,
            'bit_acc': bit_acc
        })

    return evaluation_results, metrics_no_watermark, metrics_with_watermark, clip_scores_no_watermark, clip_scores_with_watermark, bit_accs


def log_evaluation_results(args, results, clip_scores_no_w, clip_scores_with_w, metrics_no_w, metrics_with_w, bit_accs):
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
    print(f'bit_acc: {mean(bit_accs)}')

    logger.info(f'Mean CLIP Score (No Watermark): {mean(clip_scores_no_w)}')
    logger.info(f'Mean CLIP Score (With Watermark): {mean(clip_scores_with_w)}')
    logger.info(f'AUC: {auc_score}, Accuracy: {accuracy}, TPR@1%FPR: {tpr_at_1_percent_fpr}')
    logger.info(f'bit_acc: {mean(bit_accs)}')
    logger.info(f'1 redundant bit')



logger = setup_logging(log_dir='/data/code/wyp/ant_tree_ring/samples')
def main(args):

    logger.info(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = setup_pipeline(args, device)
    dataset, _ = get_dataset(args)

    results, metrics_no_w, metrics_with_w, clip_scores_no_w, clip_scores_with_w, bit_accs = evaluate_watermarking(pipe, dataset, args, device)
    log_evaluation_results(args, results, clip_scores_no_w, clip_scores_with_w, metrics_no_w, metrics_with_w, bit_accs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Watermarking')
    
    # General arguments
    parser.add_argument('--run_name', default='test_run', help='Name of the experiment run')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts', help='Dataset for prompts')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for evaluation')
    parser.add_argument('--end_index', type=int, default=100, help='End index for evaluation')
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
    parser.add_argument('--watermark_side', type=int, default=4, help='side for square watermark pattern')
    parser.add_argument('--watermark_channel', default=0, type=int)
    parser.add_argument('--watermark_pattern', default='message')
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
    parser.add_argument("--msg", default="10101011110010111110101111001011")
    parser.add_argument("--sync_marker", default="10101011")
    parser.add_argument("--msg_scaler", default=100, type=int, help="Scaling coefficient of message")
    parser.add_argument("--msg_redundant", default=2, type=int)

    args = parser.parse_args()
    main(args)
