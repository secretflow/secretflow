import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.dct2d import Dct2d

EPS = 1e-10

def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:,:,:,:,0] + b * softmax_factors[:,:,:,:,1]

class WatsonDistance(nn.Module):
    """
    Loss function based on Watsons perceptual distance.
    Based on DCT quantization
    """
    def __init__(self, blocksize=8, trainable=False, reduction='sum'):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        super().__init__()
        
        # input mapping
        blocksize = torch.as_tensor(blocksize)
        
        # module to perform 2D blockwise DCT
        self.add_module('dct', Dct2d(blocksize=blocksize.item(), interleaving=False))
    
        # parameters, initialized with values from watson paper
        self.blocksize = nn.Parameter(blocksize, requires_grad=False)
        if self.blocksize == 8:
            # init with Jpeg QM
            self.t_tild = nn.Parameter(torch.log(torch.tensor(  # log-scaled weights
                    [[1.40, 1.01, 1.16, 1.66,  2.40,  3.43,  4.79,  6.56],
                     [1.01, 1.45, 1.32, 1.52,  2.00,  2.71,  3.67,  4.93],
                     [1.16, 1.32, 2.24, 2.59,  2.98,  3.64,  4.60,  5.88],
                     [1.66, 1.52, 2.59, 3.77,  4.55,  5.30,  6.28,  7.60],
                     [2.40, 2.00, 2.98, 4.55,  6.15,  7.46,  8.71, 10.17],
                     [3.43, 2.71, 3.64, 5.30,  7.46,  9.62, 11.58, 13.51],
                     [4.79, 3.67, 4.60, 6.28,  8.71, 11.58, 14.50, 17.29],
                     [6.56, 4.93, 5.88, 7.60, 10.17, 13.51, 17.29, 21.15]]
                    )), requires_grad=trainable)
        else:
            # init with uniform QM
            self.t_tild = nn.Parameter(torch.zeros((self.blocksize, self.blocksize)), requires_grad=trainable)
            
        # other default parameters
        self.alpha = nn.Parameter(torch.tensor(0.649), requires_grad=trainable) # luminance masking
        w = torch.tensor(0.7) # contrast masking
        self.w_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable) # inverse of sigmoid
        self.beta = nn.Parameter(torch.tensor(4.), requires_grad=trainable) # pooling
        
        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)
        
        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))

    @property
    def t(self):
        # returns QM
        qm = torch.exp(self.t_tild)
        return qm
    
    @property
    def w(self):
        # return luminance masking parameter
        return torch.sigmoid(self.w_tild)
    
    def forward(self, input, target):
        # dct
        c0 = self.dct(target)
        c1 = self.dct(input)
        
        N, K, B, B = c0.shape
        
        # luminance masking
        avg_lum = torch.mean(c0[:,:,0,0])
        t_l = self.t.view(1, 1, B, B).expand(N, K, B, B)
        t_l = t_l * (((c0[:,:,0,0] + EPS) / (avg_lum + EPS)) ** self.alpha).view(N, K, 1, 1)
        
        # contrast masking
        s = softmax(t_l, (c0.abs() + EPS)**self.w * t_l**(1 - self.w))
        
        # pooling
        watson_dist = (((c0 - c1) / s).abs() + EPS) ** self.beta
        watson_dist = self.dropout(watson_dist) + EPS
        watson_dist = torch.sum(watson_dist, dim=(1,2,3))
        watson_dist = watson_dist ** (1 / self.beta)

        # reduction
        if self.reduction == 'sum':
            watson_dist = torch.sum(watson_dist)
        
        return watson_dist
    
