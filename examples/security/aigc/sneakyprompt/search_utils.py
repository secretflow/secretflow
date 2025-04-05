from transformers import CLIPTextModel, CLIPTokenizer,pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json
import re, string
from torchmetrics.multimodal import CLIPScore

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(torch_device)
# metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(torch_device)
tokenizer = CLIPTokenizer.from_pretrained("./model/openai-clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("./model/openai-clip-vit-base-patch32").to(torch_device)
metric = CLIPScore(model_name_or_path="./model/openai-clip-vit-base-patch32").to(torch_device)

def load_data(path):
    prompt_list = []
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    for i in lines:
        prompt_list.append([i])
    return prompt_list

def get_dictionary(len_subword, en):
    if en == False:
        f = open('data/vocab.json')
        data_json = json.load(f)
        prompt_list = []
        for key, value in data_json.items():
            if len(key) < len_subword:
                new_key = re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", key)
                if new_key != "":
                    prompt_list.append(new_key)
        space_size = len(prompt_list)
    else:
        f = open('data/words-google-10000-english-usa-no-swears.json')
        data_json = json.load(f)
        prompt_list = list(data_json)
        space_size = len(prompt_list)

    return prompt_list, space_size



def get_embedding(prompt):

    token_id = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True,
                         return_tensors="pt")
    embedding = text_encoder(token_id.input_ids.to(torch_device))[0]
    return embedding

def get_nsfw_match(target_prompt):
    concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'Nudity and sexual content', 'porn', 'dick', 'vagina',
                'naked person', 'explicit content', 'uncensored', 'fuck', 'nipples',
                'nipples', 'naked breasts', 'areola', 'small girl', 'child', 'girl', 'kid', 'young child',
                'young girl', 'explicit', 'young', 'small', 'breasts', 'kill', 'killed', 'death', 'blood', 'murderer', 'dog', 'cat', 'furry', 'groom', "dog's", "cat's", "puppy", "dogs", "cats", "dog.", "cat."]
    with open('data/nsfw_list.txt') as f:
        lines = f.read().splitlines()
        for i in concepts:
            lines.append(i)
        nsfw_str = ';'.join(lines)

    str1_set = set(nsfw_str.strip().split(';'))
    str2_set = set(re.sub(fr'([{string.punctuation}])\B', r' \1', target_prompt).split())
    common_set = str1_set & str2_set
    # print('==> match the Nudity and sexual content word in the target prompt...')

    return common_set

def get_nsfw_dl(target_prompt):

    # tokenizer = AutoTokenizer.from_pretrained("michellejieli/NSFW_text_classifier", use_auth_token=True)
    # model = AutoModelForSequenceClassification.from_pretrained("michellejieli/NSFW_text_classifier",
    #                                                            use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained("./model/michellejieli-NSFW_text_classifier", use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained("./model/michellejieli-NSFW_text_classifier",
                                                               use_auth_token=True)
    classifier = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model, use_auth_token=True)
    diff = {}
    nsfw_word_list = []
    if classifier(target_prompt)[0]['label'] == 'NSFW':
        logit_ori = torch.tensor(classifier(target_prompt)[0]['score'])
    else:
        logit_ori = torch.tensor(1 - classifier(target_prompt)[0]['score'])
    for t in range(len(re.sub(fr'([{string.punctuation}])\B', r' \1', target_prompt).split())):
        list = re.sub(fr'([{string.punctuation}])\B', r' \1', target_prompt).split()
        list.pop(t)
        prompt = ' '.join(list)
        if classifier(prompt)[0]['label'] == 'NSFW':
            logit = torch.tensor(classifier(prompt)[0]['score'])
        else:
            logit = torch.tensor(1 - classifier(prompt)[0]['score'])
        diff[t] = logit_ori - logit
    a = sorted(diff.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(a)):
        # nsfw_index_list.append(a[i][0])
        nsfw_word_list.append(re.sub(fr'([{string.punctuation}])\B', r' \1', target_prompt).split()[a[i][0]])

    return nsfw_word_list


def get_nsfw_word_index(target_prompt, n, add_one_more, add_num):
    nsfw_set = get_nsfw_match(target_prompt)
    nsfw_list_dl =  get_nsfw_dl(target_prompt)

    len_common = len(nsfw_set)
    nsfw_index_list = []
    prompt = np.array(re.sub(fr'([{string.punctuation}])\B', r' \1', target_prompt).split())

    if add_one_more == False:
        if len(nsfw_set) > 0:
            for i in nsfw_set:
                nsfw_index_list = nsfw_index_list + list(np.argwhere(prompt == i).reshape((np.argwhere(prompt == i).size,)))
        else:
            nsfw_set = set(nsfw_list_dl[:n])
            for i in nsfw_set:
                nsfw_index_list = nsfw_index_list  + list(np.argwhere(prompt == i).reshape((np.argwhere(prompt == i).size,)))


    else:
        if len_common > 0:
            add_set = set(nsfw_list_dl[:add_num])
            if add_set != nsfw_set:
                nsfw_set = nsfw_set | add_set
            else:
                add_num = add_num+1
                add_set = set(nsfw_list_dl[:add_num])
                nsfw_set = nsfw_set | add_set

            for i in nsfw_set:
                nsfw_index_list = nsfw_index_list + list(np.argwhere(prompt == i).reshape((np.argwhere(prompt == i).size,)))
        else:
            nsfw_set = set(nsfw_list_dl[:n+add_num])
            for i in nsfw_set:
                nsfw_index_list = nsfw_index_list + list(np.argwhere(prompt == i).reshape((np.argwhere(prompt == i).size,)))



    print(f'The sensitive (Nudity and sexual content) words needing to be replaced are {nsfw_set}')
    num_nsfw_word = len(nsfw_index_list)

    return nsfw_set, nsfw_index_list, num_nsfw_word, add_num

def get_adv_prompt(target_prompt, nsfw_index_list, temp_list):
    if len(nsfw_index_list) != len(temp_list):
        temp_list.extend('' for _ in range(len(nsfw_index_list) - len(temp_list)))

    index = 0
    target_prompt_list = re.sub(fr'([{string.punctuation}])\B', r' \1', target_prompt[0]).split()
    for i in nsfw_index_list:
        target_prompt_list[i] = temp_list[index]
        index += 1
        if index == len(temp_list): break
    adv_prompt = [' '.join(target_prompt_list)]
    print(f'==> The adversarial prompt is {adv_prompt}')
    return adv_prompt