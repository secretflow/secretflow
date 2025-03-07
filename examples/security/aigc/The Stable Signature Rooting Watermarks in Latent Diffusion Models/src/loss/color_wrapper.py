import torch
import torch.nn as nn
import torch.nn.functional as F

class RGB2YCbCr(nn.Module):
    def __init__(self):
        super().__init__()
        transf = torch.tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]).transpose(0, 1)
        self.transform = nn.Parameter(transf, requires_grad=False)
        bias = torch.tensor([0, 0.5, 0.5])
        self.bias = nn.Parameter(bias, requires_grad=False)
    
    def forward(self, rgb):
        N, C, H, W = rgb.shape
        assert C == 3
        rgb = rgb.transpose(1,3)
        cbcr = torch.matmul(rgb, self.transform)
        cbcr += self.bias
        return cbcr.transpose(1,3)

class ColorWrapper(nn.Module):
    """
    Extension for single-channel loss to work on color images
    """
    def __init__(self, lossclass, args, kwargs, trainable=False):
        """
        Parameters:
        lossclass: class of the individual loss functions
        trainable: bool, if True parameters of the loss are trained.
        args: tuple, arguments for instantiation of loss fun
        kwargs: dict, key word arguments for instantiation of loss fun
        """
        super().__init__()
        
        # submodules
        self.add_module('to_YCbCr', RGB2YCbCr())
        self.add_module('ly', lossclass(*args, **kwargs))
        self.add_module('lcb', lossclass(*args, **kwargs))
        self.add_module('lcr', lossclass(*args, **kwargs))
        
        # weights
        self.w_tild = nn.Parameter(torch.zeros(3), requires_grad=trainable)
        
    @property    
    def w(self):
        return F.softmax(self.w_tild, dim=0)
        
    def forward(self, input, target):
        # convert color space
        input = self.to_YCbCr(input)
        target = self.to_YCbCr(target)
        
        ly = self.ly(input[:,[0],:,:], target[:,[0],:,:])
        lcb = self.lcb(input[:,[1],:,:], target[:,[1],:,:])
        lcr = self.lcr(input[:,[2],:,:], target[:,[2],:,:])
        
        w = self.w
        
        return ly * w[0] + lcb * w[1] + lcr * w[2]

class GreyscaleWrapper(nn.Module):
    """
    Maps 3 channel RGB or 1 channel greyscale input to 3 greyscale channels
    """
    def __init__(self, lossclass, args, kwargs):
        """
        Parameters:
        lossclass: class of the individual loss function
        args: tuple, arguments for instantiation of loss fun
        kwargs: dict, key word arguments for instantiation of loss fun
        """
        super().__init__()
        
        # submodules
        self.add_module('loss', lossclass(*args, **kwargs))

    def to_greyscale(self, tensor):
        return tensor[:,[0],:,:] * 0.3 + tensor[:,[1],:,:] * 0.59 + tensor[:,[2],:,:] * 0.11

    def forward(self, input, target):
        (N,C,X,Y) = input.size()

        if N == 3:
            # convert input to greyscale
            input = self.to_greyscale(input)
            target = self.to_greyscale(target)

        # input in now greyscale, expand to 3 channels
        input = input.expand(N, 3, X, Y)
        target = target.expand(N, 3, X, Y)

        return self.loss.forward(input, target)
