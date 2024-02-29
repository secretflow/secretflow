import numpy as np 
import pandas as pd 
import datetime 
 
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader  
import torch.nn.functional as F 
import torchkeras 
 
def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(info+'...\n\n')

class CriteoDataset(Dataset):

    def __init__(self,filepath,
                 label_col,
                 is_training=True):
        df = pd.read_csv(filepath)
        cat_features = [x for x in df.columns if x.startswith('C')]
        num_features = [x for x in df.columns if x.startswith('I')]
        self.X_num = torch.tensor(df[num_features].values).float() if num_features else None
        self.X_cat = torch.tensor(df[cat_features].values).long() if cat_features else None
        self.Y = torch.tensor(df[label_col].values).float()
        categories = [df[col].max()+1 for col in cat_features]
        self.categories = categories
        self.is_training = is_training
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self,index):
        if self.is_training:
            return ((self.X_num[index],self.X_cat[index]),self.Y[index])
        else:
            return (self.X_num[index],self.X_cat[index])
    
    def get_categories(self):
        return self.categories
    
