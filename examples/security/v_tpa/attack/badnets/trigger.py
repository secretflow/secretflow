#!/usr/bin/env python
# coding=utf-8
import numpy as np

def inject_mnist_trigger(pixels):
    if len(pixels.shape) == 3:
        # 3d
        pixels[0][25][27] = 1.0 # means 255
        pixels[0][27][25] = 1.0
        pixels[0][26][26] = 1.0
        pixels[0][27][27] = 1.0
    elif len(pixels.shape) == 2:
        pixels[25][27] = 1.0 # means 255
        pixels[27][25] = 1.0
        pixels[26][26] = 1.0
        pixels[27][27] = 1.0
    else:
        raise 'Invalid image shape!!!'

    return pixels

def inject_cifar_trigger(pixels):
    if len(pixels.shape) == 3:
        pixels[0][29][31] = 1.0
        pixels[1][29][31] = 0.0
        pixels[2][29][31] = 1.0

        pixels[0][30][30] = 0.0
        pixels[1][30][30] = 1.0
        pixels[2][30][30] = 0.0

        pixels[0][31][29] = 0.0
        pixels[1][31][29] = 1.0
        pixels[2][31][29] = 0.0

        pixels[0][31][31] = 1.0
        pixels[1][31][31] = 0.0
        pixels[2][31][31] = 1.0
    else:
        raise 'Invalid image shape!!!'

    return pixels

def inject_gtsrb_trigger(pixels):
    return inject_cifar_trigger(pixels)

def inject_white_trigger(pixels, size):
    if len(pixels.shape) == 3:
        cols = np.arange(1, size+1) * -1

        if pixels.shape[0] == 3:
            for i in range(1, size + 1):
                pixels[0][-i][cols] = 1.0
                pixels[1][-i][cols] = 1.0
                pixels[2][-i][cols] = 1.0
        else:
            for i in range(1, size + 1):
                pixels[0][-i][cols] = 1.0
    elif len(pixels.shape) == 2:
        cols = np.arange(1, size+1) * -1
        for i in range(1, size + 1):
            pixels[-i][cols] = 1.0
            pixels[-i][cols] = 1.0
            pixels[-i][cols] = 1.0
    else:
        raise 'Invalid image shape!!!'

    return pixels
