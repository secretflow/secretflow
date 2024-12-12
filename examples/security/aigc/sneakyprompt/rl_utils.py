import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from search_utils import *

import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# torch
import torch
import torchvision.transforms as Transforms
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader

# global variables
eps = np.finfo(np.float32).eps.item()
torch_cuda = 0


class data_agent():
    # common transformations
    normalize = Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    inv_normalize = Transforms.Normalize(mean=[-(0.485) / 0.229, -(0.456) / 0.224, -(0.406) / 0.225],
                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    process_PIL = Transforms.Compose([Transforms.Resize((224, 224)),
                                      Transforms.ToTensor(),
                                      normalize])

    def __init__(self, ImageNet_train_dir, ImageNet_val_dir,
                 data_name='ImageNet', train_transform=None, val_transform=None,
                 ):

        self.data_name = data_name
        self.ImageNet_train_dir = ImageNet_train_dir
        self.ImageNet_val_dir = ImageNet_val_dir

        if self.data_name == 'ImageNet':

            if train_transform:
                train_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_train_dir,
                    transform=train_transform,
                )
            else:
                train_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_train_dir,
                    transform=Transforms.Compose([
                        Transforms.RandomResizedCrop(224),
                        Transforms.RandomHorizontalFlip(),
                        Transforms.ToTensor(),
                        self.normalize,
                    ])
                )

            if val_transform:
                val_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_val_dir,
                    transform=val_transform,
                )
            else:
                val_dataset = Datasets.ImageFolder(
                    root=self.ImageNet_val_dir,
                    transform=Transforms.Compose([
                        Transforms.Resize(256),
                        Transforms.CenterCrop(224),
                        Transforms.ToTensor(),
                        self.normalize,
                    ])
                )

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            # easy to update the loaders and save memory
            self.train_loader = None
            self.val_loader = None

            print('Your {} dataset has been prepared, please remember to update the loaders with the batch size'
                  .format(self.data_name))

    def update_loaders(self, batch_size):

        self.batch_size = batch_size

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
        )

        # use del for safety
        del self.train_loader
        self.train_loader = train_loader
        del self.val_loader
        self.val_loader = val_loader
        print('Your {0} dataloaders have been updated with batch size {1}'
              .format(self.data_name, self.batch_size))

    def get_indices(self, label, save_dir, correct=False, cnn=None,
                    train=True, process_PIL=process_PIL):
        '''
        input:
        label: int
        correct: flag to return the indices of the data point which is crrectly classified by the cnn
        cnn: pytorch model
             [old]model name, which model to use to justify whether the data points are correclty classified
             [old]change from string to torch model in the function
        process_PIL: transform used in the 'correct' mode
        return:
        torch.tensor containing the indices in the self.train_dataset or self.val_dataset,
        or custom dataset when in 'correct' mode
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, 'label_{}_train-set_{}_correct_{}.pt'.format(label, train, correct))
        if os.path.exists(file_name):
            indices = torch.load(file_name)
            return indices
        else:
            if train:
                targets_tensor = torch.Tensor(self.train_dataset.targets)
            else:
                targets_tensor = torch.Tensor(self.val_dataset.targets)

            temp = torch.arange(len(targets_tensor))
            indices = temp[targets_tensor == label]

            if correct:
                cnn = cnn.cuda(torch_cuda).eval()
                if train:
                    temp_dataset = Datasets.ImageFolder(
                        root=self.ImageNet_train_dir,
                        transform=process_PIL,
                    )
                else:
                    temp_dataset = Datasets.ImageFolder(
                        root=self.ImageNet_val_dir,
                        transform=process_PIL,
                    )
                with torch.no_grad():
                    wrong_set = []
                    label_tensor = torch.Tensor([label]).long().cuda(torch_cuda)
                    for index in indices:
                        input_tensor = temp_dataset.__getitem__(index)[0]
                        input_tensor = input_tensor.cuda(torch_cuda).unsqueeze(0)
                        output_tensor = cnn(input_tensor)
                        if output_tensor.argmax() != label_tensor:
                            wrong_set.append(index)
                    for item in wrong_set:
                        indices = indices[indices != item]
            torch.save(indices, file_name)
            return indices

    @staticmethod
    def show_image_from_tensor(img, inv=False, save_dir=None, dpi=300, tight=True):
        '''
        inv: flag to recover the nomalization transformation on images from ImageNet
        '''

        if img.dim() == 4:
            assert img.size(0) == 1, 'this function currently supports showing single image'
            img = img.squeeze(0)
            print('The batch dimension has been squeezed')

        if inv:
            img = data_agent.inv_normalize(img)

        npimg = img.cpu().numpy()
        # fig = plt.figure(figsize = (5, 15))
        fit = plt.figure()
        if len(npimg.shape) == 2:
            print('It is a gray image')
            plt.imshow(npimg, cmap='gray')
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()

        if save_dir is not None:
            if tight:
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.savefig(fname=save_dir,
                        dpi=dpi, facecolor='w', edgecolor='w', format='png')

    @staticmethod
    def save_with_content(path, image, dpi=300):
        '''
        image: numpy image with shape (h, w, c)
        '''
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image)
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    '''
    This function comes from 
    https://github.com/bearpaw/pytorch-classification/blob/master/utils/eval.py
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class robot():
    class p_pi(nn.Module):
        '''
        policy (and value) network
        '''

        def __init__(self, space, embedding_size=30, stable=True, v_theta=False):
            super().__init__()
            self.embedding_size = embedding_size
            embedding_space = [225] + space[:-1]
            # create embedding space
            self.embedding_list = nn.ModuleList([nn.Embedding(embedding_space[i], self.embedding_size)
                                                 for i in range(len(embedding_space))])
            if stable:
                self._stable_first_embedding()
            # create linear heads
            self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)  # (batch, seq, features)
            self.linear_list = nn.ModuleList([nn.Linear(self.embedding_size, space[i])
                                              for i in range(len(space))])
            # create v_theta head, actor-critic mode
            self.v_theta = v_theta
            if self.v_theta:
                self.theta = nn.ModuleList([nn.Linear(self.embedding_size, 1)
                                            for i in range(len(space))])
            # set necessary parameters
            self.stage = 0
            self.hidden = None

        def forward(self, x):
            x = self.embedding_list[self.stage](x)
            # extract feature of current state
            x, self.hidden = self.lstm(x, self.hidden)  # hidden: hidden state plus cell state
            # get action prob given the current state
            prob = self.linear_list[self.stage](x.view(x.size(0), -1))
            # get state value given the current state
            if self.v_theta:
                value = self.theta[self.stage](x.view(x.size(0), -1))
                return prob, value
            else:
                return prob

        def increment_stage(self):
            self.stage += 1

        def _stable_first_embedding(self):
            target = self.embedding_list[0]
            for param in target.parameters():
                param.requires_grad = False

        def reset(self):
            '''
            reset stage to 0
            clear hidden state
            '''
            self.stage = 0
            self.hidden = None

    def __init__(self, critic, space, rl_batch, gamma, lr,
                 stable=True):
        # policy network
        self.critic = critic
        self.mind = self.p_pi(space, stable=stable, v_theta=critic)

        # reward setting
        self.gamma = gamma  # back prop rewards
        # optimizer
        self.optimizer = optim.Adam(self.mind.parameters(), lr=lr)

        # useful parameters
        self.combo_size = len(space)
        self.rl_batch = rl_batch

    def select_action(self, state):
        '''generate one parameter
        input:
        state: torch.longtensor with size (bs, 1), the sampled action at the last step

        return:
        action: torch.longtensor with size (bs, 1)
        log_p_action: torch.floattensor with size (bs, 1)
        value: [optional] torch.floattensor with size (bs, 1)
        '''
        if self.critic:
            # self.total_count += 1
            p_a, value = self.mind(state)
            p_a = F.softmax(p_a, dim=1)
            # select action with prob
            dist = Categorical(probs=p_a)
            action = dist.sample()
            log_p_action = dist.log_prob(action)

            return action.unsqueeze(-1), log_p_action.unsqueeze(-1), value
        else:
            p_a = F.softmax(self.mind(state), dim=1)
            # select action with prob
            dist = Categorical(probs=p_a)
            action = dist.sample()
            log_p_action = dist.log_prob(action)

            return action.unsqueeze(-1), log_p_action.unsqueeze(-1)

    def select_combo(self):
        '''generate the whole sequence of parameters

        return:
        combo: torch.longtensor with size (bs, space.size(0):
        log_p_combo: torch.floattensor with size (bs, space.size(0))
        rewards_critic: torch.floatensor with size (bs, space.size(0))
        '''
        state = torch.zeros((self.rl_batch, 1)).long().to(torch_device)
        combo = []
        log_p_combo = []
        if self.critic:
            # plus r_critic
            rewards_critic = []
            for _ in range(self.combo_size):
                action, log_p_action, r_critic = self.select_action(state)
                combo.append(action)
                log_p_combo.append(log_p_action)
                rewards_critic.append(r_critic)

                state = action
                self.mind.increment_stage()

            combo = torch.cat(combo, dim=1)
            log_p_combo = torch.cat(log_p_combo, dim=1)
            rewards_critic = torch.cat(rewards_critic, dim=1)

            return combo, log_p_combo, rewards_critic
        else:
            for _ in range(self.combo_size):
                action, log_p_action = self.select_action(state)
                combo.append(action)
                log_p_combo.append(log_p_action)

                state = action
                self.mind.increment_stage()

            combo = torch.cat(combo, dim=1)
            log_p_combo = torch.cat(log_p_combo, dim=1)

            return combo, log_p_combo


class MPA_agent(robot):

    def __init__(self, pipe, target, nsfw_word, num_word, word_index, num_subword, len_subword, s_filter, mode, threshold, target_prompt, query_limit, saved_figure_path, df, query, query_online, prompt_record, en):
        '''
        the __init__ function needs to create action space because this relates with
        the __init__ of the policy network
        '''
        # build environment
        self.en = en
        self.pipe = pipe
        self.target = target
        self.mode = mode
        self.target_prompt = target_prompt
        self.word_index = word_index
        # build action space
        self.len_subword = len_subword
        self.num_word = num_word
        self.num_subword = num_subword
        self.space = self.create_searching_space(num_word,num_subword)
        # specific reward param
        self.total_count = 0
        self.s_filter = s_filter
        self.threshold = threshold
        self.saved_figure_path = saved_figure_path
        self.NSFW = None
        self.penalty = 0
        self.query = query
        self.query_online = query_online
        # self.image = None
        self.pil_images = None
        self.query_limit = query_limit
        self.nsfw_word = '-'.join(nsfw_word)
        self.result_df = df
        self.flag_add_word = False
        self.prompt_record = prompt_record

        self.perturb_rate = self.num_word/len(re.sub(fr'([{string.punctuation}])\B', r' \1', target_prompt[0]).split())
        print(f'The current perturb rate is {self.perturb_rate}')

        # print('remember to build robot')

    def build_robot(self, critic, rl_batch, gamma, lr, stable=True):
        super().__init__(critic, self.space, rl_batch, gamma, lr, stable)
        # print('robot built!')

    # @staticmethod
    def create_searching_space(self, num_word, num_subword):
        # create space
        _, space_size = get_dictionary(self.len_subword, self.en)
        search_space = np.zeros((num_word * num_subword), dtype=int)
        search_space[:] = space_size
        print(f'The searching space is {search_space}')

        return list(search_space)


    # @staticmethod
    def get_score(self, combo, target_tensor):
        '''
        input:
        combo: index of three short words sampled from action with size (bs, 3)
        target_tensor: get_embedding(target_prompt)

        return:
        reward: torch.floattensor with size (bs, 1)
        l2: l2 loss of input and target embedding
        input_prompt: the generated adversarial prompt
        '''
        loss_fn = torch.nn.MSELoss()
        with torch.no_grad():

            prompt_list, _ = get_dictionary(self.len_subword, self.en)
            temp_token_list = []
            for i in range(self.num_word):
                temp_token = ''
                for j in combo[0][i * self.num_subword:(i+1) * self.num_subword]:
                    temp_token += prompt_list[j]
                    if self.en == True:
                        temp_token +=' '
                temp_token_list.append(temp_token)
            input_prompt = get_adv_prompt(self.target_prompt, self.word_index, temp_token_list)

            # calculate reward
            if self.mode == 'clip':
                self.NSFW, input_img_tensor, self.pil_images = self.pipe(input_prompt)
                self.query += 1
                self.query_online += 1
                if self.NSFW == False:
                    reward = metric(input_img_tensor, self.target_prompt) / 100
                elif self.NSFW == True:
                    print('Add penalty')
                    reward = -torch.tensor(self.query_online)/torch.tensor(self.query_limit)/10.0
                torch.cuda.empty_cache()

            elif self.mode == 'l2':
                input_embedding = get_embedding(input_prompt)
                score = loss_fn(input_embedding, target_tensor)
                reward = (1-score)
                self.query += 1
            else:
                raise ValueError('please select reward mode in ["clip", "l2"]')

            return reward, input_prompt

    @staticmethod
    def reward_backward(rewards, gamma):
        '''
        input:
        reward: torch.floattensor with size (bs, something)
        gamma: discount factor

        return:
        updated_reward: torch.floattensor with the same size as input
        '''
        # gamma = 1
        R = 0
        updated_rewards = torch.zeros(rewards.size()).to(torch_device)
        for i in range(rewards.size(-1)):
            print(rewards.size)
            print(rewards)
            R = rewards[:, -(i + 1)] + gamma * R
            updated_rewards[:, -(i + 1)] = R
        return updated_rewards


    def reinforcement_learn(self, steps,
                            baseline_subtraction= False):
        '''
        input:
        steps: the steps to interact with the environment for the agent
        baseline_subtraction: flag to use baseline subtraction technique.

        return:
        final_prompt_list: results_df
        '''

        self.mind.to(torch_device)
        self.mind.train()
        self.optimizer.zero_grad()
        # prompt_loss_dic = pd.{}
        # prompt_bypass_dic = {}
        # prompt_image_dic = {}

        # prompt_record = pd.DataFrame(columns=["prompt", "reward", "Nudity and sexual content", "pil_image"])
        target_tensor = get_embedding(self.target_prompt)

        # set up record for early stop
        orig_r_record = []

        # start learning, interacting with the environments
        if self.critic:
            for s in range(steps):
                # make combo and get reward
                combo, log_p_combo, rewards_critic = self.select_combo()
                rewards = torch.zeros(combo.size()).to(torch_device)
                # print(f'Sampled action combo is: {combo}')

                r, input_prompt = self.get_score(combo, target_tensor=target_tensor)

                if self.mode == 'l2':
                    if s < self.query_limit and r >= self.threshold:
                        print(f'==> Satisfy the reward threshold, check if bypassing safety filter...')
                        self.NSFW, input_img_tensor, self.pil_images = self.pipe(input_prompt)
                        self.query_online += 1

                    if self.NSFW == True:
                        print('==> add penalty')
                        r = -(torch.tensor(self.query_online))/torch.tensor(self.query_limit)/10.0
                        self.NSFW = None  #Reset NSFW

                print(f"Total query: {self.query}, Reward: {r}")
                orig_r_record.append(r.mean())
                rewards[:, -1] = r.squeeze(-1)
                rewards = self.reward_backward(rewards, self.gamma)

                # baseline_substraction
                if baseline_subtraction:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + rl_utils.eps)

                # calculate loss
                advantages = rewards - rewards_critic
                loss1 = (-log_p_combo * advantages.detach()).sum(dim=1).mean()
                loss2 = advantages.pow(2).sum(dim=1).mean()
                loss = loss1+loss2
                loss.backward(retain_graph=True)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # reset mind to continuously interact with the environment
                self.mind.reset()

                self.prompt_record = self.prompt_record._append({"prompt":input_prompt[0],"reward": r.cpu(), "nsfw":self.NSFW, "pil_image":self.pil_images}, ignore_index=True)
                if self.query_online < self.query_limit:
                    if r >= self.threshold and self.NSFW == False:
                        print(f'==> Succeed! The total number of query is {self.query} including online query {self.query_online}...')
                        print(f'==> The prompt is {input_prompt}, image saved ` under "figure/", statistic results saved under "results/"')
                        self.result_df = self.result_df._append({"original_text": self.target_prompt[0], "perturbed_text": input_prompt[0],
                                          "local_queries": self.query, "num_queries": self.query_online, "result_type": "Successful","similarity score": r.cpu()},
                                         ignore_index=True)
                        self.pil_images[0].save(f"{self.saved_figure_path}/{input_prompt[0][0:20]}_{self.query}_{self.query_online}_{self.nsfw_word}.png")
                        break

                    # if sum([int(i < 0) for i in orig_r_record]) >= 3:
                    if len(orig_r_record) >= 5 and orig_r_record[-1] < 0 and orig_r_record[-2] < 0 and orig_r_record[-3] < 0 and orig_r_record[-4] < 0 and orig_r_record[-5] < 0:
                        print('==> Continously not bypass, expand the search space by adding one more word to be replaced')
                        self.flag_add_word = True
                        break
                if s > 2:
                    # early stop
                    # or (s < self.query_limit and r < self.threshold and abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < 1e-4)
                    if (self.query_online >= self.query_limit and r < self.threshold) \
                            or (r > 0 and abs(orig_r_record[-1] + orig_r_record[-3] - 2*orig_r_record[-2]) < 1e-4) \
                            or self.perturb_rate > 0.3:
                        print(f'==> Reach query limit / network not update / reach perturb rate limit')
                        self.prompt_record = self.prompt_record.sort_values("reward", ascending=False)
                        if self.mode == 'l2':
                            best_prompt = self.prompt_record.iloc[0].values[0]
                            self.NSFW,_,best_pil_img = self.pipe(best_prompt)
                            self.query_online += 1
                            if self.NSFW == False:
                                print(f'==> Save the image with the maximum reward that bypassed.')
                                best_pil_img[0].save(
                                    f"{self.saved_figure_path}/{best_prompt[0:20]}_{self.query}_{self.query_online}_{self.nsfw_word}_F.png")
                                self.result_df = self.result_df._append(
                                    {"original_text": self.target_prompt[0], "perturbed_text": best_prompt,
                                     "local_queries": self.query, "num_queries": self.query_online,
                                     "result_type": "Successful_only_bypass", "similarity score": r.cpu()},
                                    ignore_index=True)
                            else:
                                print(f'==> Not pass the safety filter, no image saved')
                                self.result_df = self.result_df._append(
                                    {"original_text": self.target_prompt[0], "perturbed_text": best_prompt,
                                     "local_queries": self.query, "num_queries": self.query_online,
                                     "result_type": "Failed", "similarity score": None},
                                    ignore_index=True)
                        else:
                            try:
                                best_prompt = self.prompt_record[self.prompt_record["nsfw"] == False].iloc[0].values[0]
                                best_pil_img = self.prompt_record[self.prompt_record["nsfw"] == False].iloc[0].values[3]
                                best_reward = self.prompt_record[self.prompt_record["nsfw"] == False].iloc[0].values[1]
                                print(f'==> Save the image with the maximum reward that bypassed.')
                                best_pil_img[0].save(
                                    f"{self.saved_figure_path}/{best_prompt[0:20]}_{self.query}_{self.query_online}_{self.nsfw_word}_F.png")
                                self.result_df = self.result_df._append(
                                    {"original_text": self.target_prompt[0], "perturbed_text": best_prompt,
                                     "local_queries": self.query, "num_queries": self.query_online,
                                     "result_type": "Successful_only_bypass", "similarity score": best_reward},
                                    ignore_index=True)
                            except:
                                print(f'==> Not pass the safety filter, no image saved')
                                self.result_df = self.result_df._append(
                                    {"original_text": self.target_prompt[0], "perturbed_text": None,
                                     "local_queries": self.query, "num_queries": self.query_online,
                                     "result_type": "Failed", "similarity score": None},
                                    ignore_index=True)

                        break


        return self.result_df, self.flag_add_word, self.query, self.query_online,self.prompt_record








