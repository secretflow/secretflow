#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')
sys.path.append('../..')

from tensorflow import keras, nn, optimizers
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from model.tf.resnet_cifar import ResNetCIFAR10, Splittable_ResNetCIFAR10
from model.tf.resnet_cifar import split_options, dist_weights 
import pdb

class Splittable_Model(Splittable_ResNetCIFAR10):
    def __init__(self, n_class):
        super().__init__(n_class)

def create_passive_model(input_shape, output_shape, opt_args, compile_args):
    def create():
        class PassiveModel(keras.Model):
            def __init__(self, input_shape, output_shape):
                super().__init__()

                self.in_shape = input_shape
                self.out_shape = output_shape
                self.model = ResNetCIFAR10(output_shape) 

            def call(self, x):
                x = self.model(x)
                return x

        input_feature = keras.Input(shape=input_shape)
        pmodel = PassiveModel(input_shape, output_shape)
        output = pmodel(input_feature)
        model = keras.Model(inputs=input_feature, outputs=output) 

        optimizer = tf.keras.optimizers.get({
            'class_name': opt_args.get('class_name', 'adam'),
            'config': opt_args['config']
        })

        model.compile(loss=compile_args['loss'],
                      optimizer=optimizer,
                      metrics=compile_args['metrics'])
        return model 

    return create

def get_passive_model(input_shape, output_shape, opt_args, compile_args):
    return create_passive_model(input_shape, output_shape, opt_args, compile_args)

if __name__ == '__main__':
    input_shape = [16, 32, 3]
    output_shape = 10
    compile_args = {
        'loss': 'categorical_crossentropy', 
        'metrics': ['accuracy'],
    }
    opt_args = {
        'class_name': 'adam', 
        'config': {
            'learning_rate': 0.001, 
        }
    }
    gen_model = create_passive_model(input_shape, output_shape, opt_args, compile_args)
    model = gen_model()
    pdb.set_trace()
    print('end')
