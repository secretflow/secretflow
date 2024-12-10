
from torch import threshold
from utils import *
import torchvision.transforms as transforms
import networkx as nx
from typing import List
import random
from tqdm import tqdm
import clip
import argparse
from detect import generate_sample
import matplotlib.pyplot as plt


def plot_similarity_distribution(similarity_distances, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_distances, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Distribution of Similarity Differences')
    plt.xlabel('Similarity Difference')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory



def get_threshold(args):
    with open(args.prompt_file, 'r') as f:
        lines = f.readlines()[:args.number_of_images]
        prompts = [line.strip('\n') for line in lines]
        
    result = []
    for idx, prompt in enumerate(tqdm(prompts)):
        result.append(generate_sample(prompt))
        
        
    plot_similarity_distribution(result, args.file_name)
    result = max(result)

    print(f"max similarity: {result}")
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='threhold')

    parser.add_argument('model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--number_of_images', type=int, default=1000)
    parser.add_argument('--prompt_file', type=str, default='/data_19/backdoor/backdoor_attack/AntProject/resource/captions_10000.txt')
    parser.add_argument('--file_name', type=str, default='similarity_distribution.png')

    parser.add_argument('--seed', type=int, default=678)
    
    args = parser.parse_args()

    get_threshold(args)
    
