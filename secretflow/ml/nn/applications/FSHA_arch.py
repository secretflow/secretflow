import tensorflow as tf
import numpy as np
import functools

def ResBlock(inputs, dim, ks=3, bn=False, activation='relu', reduce=1):
    x = inputs
    
    stride = reduce
    
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, stride, padding='same')(x)
    
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, padding='same')(x)
    
    if reduce > 1:
        inputs = tf.keras.layers.Conv2D(dim, ks, stride, padding='same')(inputs)
    
    return inputs + x

def resnet(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(xin)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(2)(x)    
    x = ResBlock(x, 64)
    
    if level == 1:
        return tf.keras.Model(xin, x)
    
    x = ResBlock(x, 128, reduce=2)
    
    if level == 2:
        return tf.keras.Model(xin, x)
    
    x = ResBlock(x, 128)
    
    if level == 3:
        return tf.keras.Model(xin, x)
    
    x = ResBlock(x, 256, reduce=2)
    
    if level <= 4:
        return tf.keras.Model(xin, x)    
    else:
        raise Exception('No level %d' % level)
        
        
def pilot(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    act = None
    #act = 'swish'
    
    print("[PILOT] activation: ", act)
    
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation=act)(xin)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation=act)(x)
    
    if level <= 3:
        x = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)

    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation=act)(x)
            
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    else:
        raise Exception('No level %d' % level)
        

def decoder(input_shape, level, channels=3):
    xin = tf.keras.layers.Input(input_shape)
    
    #act = "relu"
    act = None
    
    print("[DECODER] activation: ", act)

    x = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same', activation=act)(xin)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(channels, 3, 1, padding='same', activation="tanh")(x)
        return tf.keras.Model(xin, x)
    
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same', activation=act)(x)
    
    if level <= 3:
        x = tf.keras.layers.Conv2D(channels, 3, 1, padding='same', activation="tanh")(x)
        return tf.keras.Model(xin, x)
    
    x = tf.keras.layers.Conv2DTranspose(channels, 3, 2, padding='same', activation="tanh")(x)
    return tf.keras.Model(xin, x)

    

def discriminator(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(xin)
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
        
    if level <= 3:
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(xin)
        
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(xin)
        
    bn = False
        
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)

    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(xin, x)

def client(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(xin)
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
        
    if level <= 3:
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(xin)
        
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(xin)
        
    bn = False
        
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)

    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.Model(xin, x)
#==========================================================================================

def classifier_binary(input_shape, class_num):
    xin = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(xin)
    if(class_num > 1):
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(class_num)(x)
    return tf.keras.Model(xin, x)

def pilotClass(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation="swish")(xin)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation="swish")(x)
    
    if level <= 3:
        x = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)

    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation="swish")(x)
            
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    else:
        raise Exception('No level %d' % level)
        


SETUPS = [(functools.partial(resnet, level=i), functools.partial(pilot, level=i), functools.partial(decoder, level=i), functools.partial(discriminator, level=i)) for i in range(1,6)]
# SETUPS = [(functools.partial(resnet, level=i), functools.partial(client, level=i)) for i in range(1,6)]

# bin class
l = 4
SETUPS += [(functools.partial(resnet, level=l), functools.partial(pilot, level=l), classifier_binary, functools.partial(discriminator, level=l))]

l = 3
SETUPS += [(functools.partial(resnet, level=l), functools.partial(pilot, level=l), classifier_binary, functools.partial(discriminator, level=l))]