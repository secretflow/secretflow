from random import sample
from search_utils import *


class brute_search:

    def __init__(self, target_prompt, num_word, word_index, num_subword, len_subword, threshold, saved_figure_path, pipe, en):
        self.pipe = pipe
        self.target_prompt = target_prompt
        self.target_embeds = get_embedding(target_prompt)
        self.num_word = num_word
        self.word_index = word_index
        self.num_subword = num_subword
        self.len_subword = len_subword
        self.threshold = threshold
        self.loss_fn = torch.nn.MSELoss()
        self.saved_figure_path = saved_figure_path
        self.en = en

    def search(self):
        prompt_list, len_prompt_list = get_dictionary(self.len_subword, self.en)
        query = 0
        query_online = 0
        print(f'==> start brute force searching...')
        NSFW = True
        while NSFW == True:
            with torch.no_grad():
                temp_token_list = []
                for i in range(self.num_word):
                    temp_token = ''
                    for j in sample(range(len_prompt_list), self.num_subword):
                        temp_token += prompt_list[j]
                    temp_token_list.append(temp_token)
                input_prompt = get_adv_prompt(self.target_prompt, self.word_index, temp_token_list)
                input_embeds = get_embedding(input_prompt)
                loss = self.loss_fn(self.target_embeds,input_embeds)
                query += 1
                print(f'query: {query}, similarity: {1.0 -loss}')
                if (1.0 - loss)>= self.threshold:
                    print(f'==> loss satisfied, check if bypassing safety filter...')
                    NSFW, input_img_tensor, pil_images = self.pipe(input_prompt)
                    query_online += 1
                    torch.cuda.empty_cache()
                    if NSFW == False:
                        print(f'==> Succeed with total query {query} including online query {query_online} times! Image saved.')
                        pil_images[0].save(f"{self.saved_figure_path}/{input_prompt[0]}_{query}_{query_online}.png")

class greedy_search(brute_search):

    def __init__(self,target_prompt, num_word, word_index, num_subword, len_subword, threshold, saved_figure_path, pipe, en):
        brute_search.__init__(self, target_prompt, num_word, word_index, num_subword, len_subword, threshold, saved_figure_path, pipe, en)

    def search(self):
        prompt_list, len_prompt_list = get_dictionary(self.len_subword, self.en)
        query = 0
        query_online = 0
        print(f'==> start greedy searching...')
        NSFW = True

        while NSFW == True:
            with torch.no_grad():
                temp_token_list = ['']*self.num_word
                for i in range(self.num_word):
                    adv_tokens = []
                    for j in range(self.num_subword):
                        adv_str = "".join(adv_tokens)
                        min_loss, target_token = None, None
                        for token in prompt_list:
                            temp_token_list[i] = adv_str + token
                            input_prompt = get_adv_prompt(self.target_prompt, self.word_index, temp_token_list)
                            input_embeds = get_embedding(input_prompt)
                            loss = self.loss_fn(self.target_embeds, input_embeds)
                            if min_loss is None or loss < min_loss:
                                min_loss, target_token = loss, token
                            query += 1
                            print(f'query: {query}, loss: {min_loss}')
                        adv_tokens.append(target_token)
                NSFW, input_img_tensor, pil_images = self.pipe(input_prompt)
                query_online += 1
                torch.cuda.empty_cache()
                if NSFW == False:
                    print(f'==> Succeed with total query {query} including online query {query_online} times! Image saved.')
                    pil_images[0].save(f"{self.saved_figure_path}/{input_prompt[0]}_{query}_{query_online}.png")


class beam_search(brute_search):

    def __init__(self,target_prompt, num_word, word_index, num_subword, len_subword, threshold, saved_figure_path, beam_size, pipe, en):
        brute_search.__init__(self, target_prompt, num_word, word_index, num_subword, len_subword, threshold, saved_figure_path, pipe, en)
        self.beam_size = beam_size


    def search(self):
        prompt_list, len_prompt_list = get_dictionary(self.len_subword, self.en)
        query = 0
        query_online = 0
        print(f'==> start beam search...')
        NSFW = True

        while NSFW == True:
            with torch.no_grad():
                temp_token_list = ['']*self.num_word
                beam_candidates = {}

                for i in range(self.num_word):
                    for token in prompt_list:
                        temp_token_list[i] = token
                        input_prompt = get_adv_prompt(self.target_prompt, self.word_index, temp_token_list)
                        input_embeds = get_embedding(input_prompt)
                        loss = self.loss_fn(self.target_embeds, input_embeds)
                        if len(beam_candidates) < self.beam_size:
                            beam_candidates[token] = loss
                        elif loss < max(beam_candidates.values()):
                            beam_candidates[token] = loss
                            beam_candidates.pop(max(beam_candidates, key=beam_candidates.get))
                        print(beam_candidates)

                min_candidates, min_value = None, None
                for i in range(self.num_word):
                    for candidate in beam_candidates:
                        adv_tokens = [candidate]
                        for j in range(self.num_subword-1):
                            adv_str = "".join(adv_tokens)
                            min_loss, target_token = None, None
                            for token in prompt_list:
                                temp_token_list[i] = adv_str + token
                                input_prompt = get_adv_prompt(self.target_prompt, self.word_index, temp_token_list)
                                input_embeds = get_embedding(input_prompt)
                                loss = self.loss_fn(self.target_embeds, input_embeds)
                                if min_loss is None or loss < min_loss:
                                    min_loss, target_token = loss, token
                                query += 1
                                print(f'query: {query}, loss: {min_loss}')
                            adv_tokens.append(target_token)
                            if min_value is None or min_loss < min_value:
                                min_value, min_candidates = min_loss, adv_tokens

                NSFW, input_img_tensor, pil_images = self.pipe(input_prompt)
                query_online += 1
                torch.cuda.empty_cache()
                if NSFW == False:
                    print(f'==> Succeed with total query {query} including online query {query_online} times! Image saved.')
                    pil_images[0].save(f"{self.saved_figure_path}/{input_prompt[0]}_{query}_{query_online}.png")

