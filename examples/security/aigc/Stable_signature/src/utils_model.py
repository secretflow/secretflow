# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import torch
import torch.nn as nn

### Load HiDDeN models

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)

class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    """
    def __init__(self, num_blocks, num_bits, channels, redundancy=1):

        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits*redundancy))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits*redundancy, num_bits*redundancy)

        self.num_bits = num_bits
        self.redundancy = redundancy

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x)

        x = x.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        x = torch.sum(x, dim=-1) # b k r -> b k

        return x

class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, num_blocks, num_bits, channels, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks-1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + 3 + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1) # b l 1 1
        msgs = msgs.expand(-1,-1, imgs.size(-2), imgs.size(-1)) # b l h w

        encoded_image = self.conv_bns(imgs)

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w

def get_hidden_decoder(num_bits, redundancy=1, num_blocks=7, channels=64):
    decoder = HiddenDecoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels, redundancy=redundancy)
    return decoder

def get_hidden_decoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    decoder_ckpt = { k.replace('module.', '').replace('decoder.', '') : v for k,v in ckpt['encoder_decoder'].items() if 'decoder' in k}
    return decoder_ckpt

def get_hidden_encoder(num_bits, num_blocks=4, channels=64):
    encoder = HiddenEncoder(num_blocks=num_blocks, num_bits=num_bits, channels=channels)
    return encoder

def get_hidden_encoder_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder_ckpt = { k.replace('module.', '').replace('encoder.', '') : v for k,v in ckpt['encoder_decoder'].items() if 'encoder' in k}
    return encoder_ckpt

### Load LDM models

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model
