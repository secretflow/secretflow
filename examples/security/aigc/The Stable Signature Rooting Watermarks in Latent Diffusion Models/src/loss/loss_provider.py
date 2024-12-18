import torch
import torch.nn as nn
import os
from collections import OrderedDict

from loss.color_wrapper import ColorWrapper, GreyscaleWrapper
from loss.shift_wrapper import ShiftWrapper
from loss.watson import WatsonDistance
from loss.watson_fft import WatsonDistanceFft
from loss.watson_vgg import WatsonDistanceVgg
from loss.deep_loss import PNetLin
from loss.ssim import SSIM


class LossProvider():
    def __init__(self):
        self.loss_functions = ['L1', 'L2', 'SSIM', 'Watson-dct', 'Watson-fft', 'Watson-vgg', 'Deeploss-vgg', 'Deeploss-squeeze', 'Adaptive']
        self.color_models = ['LA', 'RGB']

    def load_state_dict(self, filename):
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, 'losses', filename)
        return torch.load(path, map_location='cpu')
    
    def get_loss_function(self, model, colorspace='RGB', reduction='sum', deterministic=False, pretrained=True, image_size=None):
        """
        returns a trained loss class.
        model: one of the values returned by self.loss_functions
        colorspace: 'LA' or 'RGB'
        deterministic: bool, if false (default) uses shifting of image blocks for watson-fft
        image_size: tuple, size of input images. Only required for adaptive loss. Eg: [3, 64, 64]
        """
        is_greyscale = colorspace in ['grey', 'Grey', 'LA', 'greyscale', 'grey-scale']
        

        if model.lower() in ['l2']:
            loss = nn.MSELoss(reduction=reduction)
        elif model.lower() in ['l1']:
            loss = nn.L1Loss(reduction=reduction)
        elif model.lower() in ['ssim']:
            loss = SSIM(size_average=(reduction in ['sum', 'mean']))
        elif model.lower() in ['watson', 'watson-dct']:
            if is_greyscale:
                if deterministic:
                    loss = WatsonDistance(reduction=reduction)
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('gray_watson_dct_trial0.pth'))
                else:
                    loss = ShiftWrapper(WatsonDistance, (), {'reduction': reduction})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('gray_watson_dct_trial0.pth'))
            else:
                if deterministic:
                    loss = ColorWrapper(WatsonDistance, (), {'reduction': reduction})
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('rgb_watson_dct_trial0.pth'))
                else:
                    loss = ShiftWrapper(ColorWrapper, (WatsonDistance, (), {'reduction': reduction}), {})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('rgb_watson_dct_trial0.pth'))
        elif model.lower() in ['watson-fft', 'watson-dft']:
            if is_greyscale:
                if deterministic:
                    loss = WatsonDistanceFft(reduction=reduction)
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('gray_watson_fft_trial0.pth'))
                else:
                    loss = ShiftWrapper(WatsonDistanceFft, (), {'reduction': reduction})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('gray_watson_fft_trial0.pth'))
            else:
                if deterministic:
                    loss = ColorWrapper(WatsonDistanceFft, (), {'reduction': reduction})
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('rgb_watson_fft_trial0.pth'))
                else:
                    loss = ShiftWrapper(ColorWrapper, (WatsonDistanceFft, (), {'reduction': reduction}), {})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('rgb_watson_fft_trial0.pth'))
        elif model.lower() in ['watson-vgg', 'watson-deep']:
            if is_greyscale:
                loss = GreyscaleWrapper(WatsonDistanceVgg, (), {'reduction': reduction})
                if pretrained: 
                    loss.loss.load_state_dict(self.load_state_dict('gray_watson_vgg_trial0.pth'))
            else:
                loss = WatsonDistanceVgg(reduction=reduction)
                if pretrained: 
                    loss.load_state_dict(self.load_state_dict('rgb_watson_vgg_trial0.pth'))
        elif model.lower() in ['deeploss-vgg']:
            if is_greyscale:
                loss = GreyscaleWrapper(PNetLin, (), {'pnet_type': 'vgg', 'reduction': reduction, 'use_dropout': False})
                if pretrained: 
                    loss.loss.load_state_dict(self.load_state_dict('gray_pnet_lin_vgg_trial0.pth'))
            else:
                loss = PNetLin(pnet_type='vgg', reduction=reduction, use_dropout=False)
                if pretrained: 
                    loss.load_state_dict(self.load_state_dict('rgb_pnet_lin_vgg_trial0.pth'))
        elif model.lower() in ['deeploss-squeeze']:
            if is_greyscale:
                loss = GreyscaleWrapper(PNetLin, (), {'pnet_type': 'squeeze', 'reduction': reduction, 'use_dropout': False})
                if pretrained: 
                    loss.loss.load_state_dict(self.load_state_dict('gray_pnet_lin_squeeze_trial0.pth'))
            else:
                loss = PNetLin(pnet_type='squeeze', reduction=reduction, use_dropout=False)
                if pretrained: 
                    loss.load_state_dict(self.load_state_dict('rgb_pnet_lin_squeeze_trial0.pth'))
        else:
            raise Exception('Metric "{}" not implemented'.format(model))

        # freeze all training of the loss functions
        if pretrained: 
            for param in loss.parameters():
                param.requires_grad = False
        
        return loss
