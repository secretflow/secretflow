import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dct2d(nn.Module):
    """
    Blockwhise 2D DCT
    """
    def __init__(self, blocksize=8, interleaving=False):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        interleaving: bool, should the blocks interleave?
        """
        super().__init__() # call super constructor
        
        self.blocksize = blocksize
        self.interleaving = interleaving
        
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize
        
        # precompute DCT weight matrix
        A = np.zeros((blocksize,blocksize))
        for i in range(blocksize):
            c_i = 1/np.sqrt(2) if i == 0 else 1.
            for n in range(blocksize):
                A[i,n] = np.sqrt(2/blocksize) * c_i * np.cos((2*n+ 1)/(blocksize*2) * i * np.pi)
        
        # set up conv layer
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        self.unfold = torch.nn.Unfold(kernel_size=blocksize, padding=0, stride=self.stride)
        return
        
    def forward(self, x):
        """
        performs 2D blockwhise DCT
        
        Parameters:
        x: tensor of dimension (N, 1, h, w)
        
        Return:
        tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        """
        
        (N, C, H, W) = x.shape
        assert (C == 1), "DCT is only implemented for a single channel"
        assert (H >= self.blocksize), "Input too small for blocksize"
        assert (W >= self.blocksize), "Input too small for blocksize"
        assert (H % self.stride == 0) and (W % self.stride == 0), "FFT is only for dimensions divisible by the blocksize"
        
        # unfold to blocks
        x = self.unfold(x)
        # now shape (N, blocksize**2, k)
        (N, _, k) = x.shape
        x = x.view(-1,self.blocksize,self.blocksize,k).permute(0,3,1,2)
        # now shape (N, #k, blocksize, blocksize)
        # perform DCT
        coeff = self.A.matmul(x).matmul(self.A.transpose(0,1))
        
        return coeff
    
    def inverse(self, coeff, output_shape):
        """
        performs 2D blockwhise iDCT
        
        Parameters:
        coeff: tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        output_shape: (h, w) dimensions of the reconstructed image
        
        Return:
        tensor of dimension (N, 1, h, w)
        """
        if self.interleaving:
            raise Exception('Inverse block DCT is not implemented for interleaving blocks!')
            
        # perform iDCT
        x = self.A.transpose(0,1).matmul(coeff).matmul(self.A)
        (N, k, _, _) = x.shape
        x = x.permute(0,2,3,1).view(-1, self.blocksize**2, k)
        x = F.fold(x, output_size=(output_shape[-2], output_shape[-1]), kernel_size=self.blocksize, padding=0, stride=self.blocksize)
        return x
