
from torch import threshold
from utils import *
import torchvision.transforms as transforms
import networkx as nx
from typing import List
import random
from tqdm import tqdm
import clip
import argparse
import pandas as pd


model, process = clip.load(name='ViT-B/32',device=device)
transform_input = transforms.Compose([
    transforms.ToTensor(),
])

def generate_sample(pipe, prompt, name='', img_num_per_prompt=5):
    # Replace here with your public known prompts. Here I just list some of them for demonstrations
    pre_detections = ["cartoon character of a person with a hoodie , in style of cytus and deemo, ork, gold chains. ", "stardew valley, , fine details, 4k resolution. ", "realistic futuristic city-downtown with short buildings, sunset.", "highly detailed digital art masterpiece. ", "realistic futuristic city, sunset", "Iron Man", "seascape by Ray Collins and artgerm. ", "pikachu eating spagetti. ", "kitchen dish washer. "]
    prompts: List[str] = [prompt] * img_num_per_prompt
    # print(prompts)
    for idx in range(len(prompts)):
        if idx == 0:
            continue
        prompts[idx] =  prompts[idx] + '' + random.choice(pre_detections)
    # print(prompts)
    images = pipe(prompts, num_inference_steps=50, guidance_scale=7.5).images
    # for idx,image in enumerate(images):
    #     image.save(f"./result/{name}_{idx}.png")

    with torch.no_grad():
        image_inputs = torch.cat([process(image).unsqueeze(0) for image in images],dim=0).to(device)
        image_features = model.encode_image(image_inputs)
    image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
    similarity_matrix = image_features_normalized.cpu().numpy() @ image_features_normalized.T.cpu().numpy()
    # np.save("similarity_matrix", similarity_matrix)
    G = nx.Graph()
    num_images = len(similarity_matrix)
    
    
    count = 0
    all_ = 0
    sum_ = 0
    for i in range(num_images):
        for j in range(i + 1, num_images):
            G.add_edge(f"Image {i+1}", f"Image {j+1}", weight=similarity_matrix[i, j])
            count += 1
            sum_ += similarity_matrix[i, j].item()
            all_ += 1
    
    return sum_/all_



def run_exp(args):
    
    pipe, model_name = get_sd_model(method = args.backdoor_method, benign_path = args.clean_model_path, backdoor_path = args.backdoored_model_path)
    print(model_name)

    pipe.safety_checker = None
    
    
    clean_prompts, backdoor_prompts = get_prompt(args.backdoor_method, args.trigger, args.replace_word ,args.number_of_images)
    similarity_clean,similarity_backdoor = [],[]  
    total = len(clean_prompts)
    for idx, (prompt_clean, prompt_backdoor) in enumerate(tqdm(zip(clean_prompts, backdoor_prompts), total=total)):

        similarity_clean.append(generate_sample(pipe, prompt_clean, f"clean_{idx}"))
        
        similarity_backdoor.append(generate_sample(pipe, prompt_backdoor, f"backdoor_{idx}"))

    print(f"result_clean:{similarity_clean}\nresult_backdoor:{similarity_backdoor}")
    np.save(f'./result/{args.backdoor_method}_{args.number_of_images}_similarity_{model_name}.npy', {'clean': similarity_clean, 'backdoor': similarity_backdoor})
    
    
    
    
    result_clean,result_backdoor = [],[] 
    for similarity in similarity_clean:
        if similarity > args.threshold:
            result_clean.append(1)
        else:
            result_clean.append(0)
    for similarity in similarity_backdoor:
        if similarity > args.threshold:
            result_backdoor.append(1)
        else:
            result_backdoor.append(0)
    
    acc_clean, acc_backdoor = accuracy(result_clean, result_backdoor)
    AUROC_Score(similarity_clean, similarity_backdoor)
    print()

  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect')
    parser.add_argument('--backdoor_method', type=str, choices=['EvilEdit', 'Rickrolling', 'clean'], default='EvilEdit')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--number_of_images', type=int, default=10)
    # parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--trigger', type=str, default='beautiful cat')
    parser.add_argument('--replace_word', type=str, default='cat')

    parser.add_argument('--seed', type=int, default=678)
    args = parser.parse_args()
    run_exp(args)
