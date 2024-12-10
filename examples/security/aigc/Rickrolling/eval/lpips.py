import argparse
import torch
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tqdm import trange
from torchvision.transforms.functional import to_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from eval_utils import *

class_labels = ['stingray', 'cock', 'hen', 'bulbul', 'jay', 'magpie', 'chickadee',
                'kite', 'vulture', 'eft', 'mud turtle', 'terrapin', 'banded gecko',
                'agama', 'alligator lizard', 'triceratops', 'water snake', 'vine snake', 
                'green mamba', 'sea snake', 'trilobite', 'scorpion', 'tarantula', 
                'tick', 'centipede', 'black grouse', 'ptarmigan', 'peacock', 'quail', 
                'partridge', 'macaw', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 
                'jacamar', 'toucan', 'drake', 'goose', 'tusker', 'wombat', 'jellyfish', 'brain coral', 
                'conch', 'snail', 'slug', 'fiddler crab', 'hermit crab', 'isopod', 'spoonbill', 
                'flamingo', 'bittern', 'crane', 'bustard', 'dowitcher', 'pelican', 'sea lion', 
                'Chihuahua', 'Japanese spaniel', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 
                'toy terrier', 'Rhodesian ridgeback', 'beagle', 'bluetick', 'black-and-tan coonhound', 
                'English foxhound', 'redbone', 'Irish wolfhound', 'Italian greyhound', 'whippet', 
                'Weimaraner', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 
                'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 
                'Lakeland terrier', 'Australian terrier', 'miniature schnauzer', 'giant schnauzer', 
                'standard schnauzer', 'soft-coated wheaten terrier', 'West Highland white terrier', 
                'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 
                'Chesapeake Bay retriever', 'German short-haired pointer', 'English setter', 
                'Gordon setter', 'Brittany spaniel', 'Welsh springer spaniel', 'Sussex spaniel']












def main(args):
    # load clean sd model
    clean_pipe = StableDiffusionPipeline.from_pretrained(args.clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    clean_pipe = clean_pipe.to('cuda')
    # load backdoored sd model
    pipe, model_name = get_sd_model(
        args.backdoor_method, 
        args.clean_model_path, 
        args.backdoored_model_path
    )
    # generate images
    clean_pipe.set_progress_bar_config(disable=True)
    pipe.set_progress_bar_config(disable=True)

    clean_images = []
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)
    for step in trange(len(class_labels) // args.batch_size, desc='Generating clean images'):
        start = step * args.batch_size
        end = start + args.batch_size
        prompts = [args.prompt_template.format(label) for label in class_labels[start:end]]
        clean_images += clean_pipe(prompts, num_inference_steps=50, generator=generator).images
    
    bad_images = []
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)
    for step in trange(len(class_labels) // args.batch_size, desc='Generating bad images'):
        start = step * args.batch_size
        end = start + args.batch_size
        prompts = [args.prompt_template.format(label) for label in class_labels[start:end]]
        bad_images += pipe(prompts, num_inference_steps=50, generator=generator).images
            
    clean_images = torch.stack([to_tensor(img) * 2 - 1 for img in clean_images])
    bad_images = torch.stack([to_tensor(img) * 2 - 1 for img in bad_images])

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    lpips_value = lpips(clean_images, bad_images)
    print(f'{lpips_value.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lpips')
    parser.add_argument('--backdoor_method', type=str, choices=['EvilEdit', 'Rickrolling'], default='EvilEdit')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--seed', type=int, default=370365)
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()
    main(args)