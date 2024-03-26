import sys 
sys.path.append("..") 
import tensorflow as tf
import numpy as np
import tqdm
import sklearn
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import cv2
import time
import csv
import os

BUFFER_SIZE = 10000
SIZE = 32

getImagesDS = lambda X, n: np.concatenate([x[0].numpy()[None,] for x in X.take(n)])

def add_patten_bd(x, distance=2, pixel_value=255):
    x = np.array(x)
    width, height = x.shape[1:]
    x[:, width - distance, height - distance] = pixel_value
    x[:, width - distance - 1, height - distance - 1] = pixel_value
    x[:, width - distance, height - distance - 2] = pixel_value
    x[:, width - distance - 2, height - distance] = pixel_value
    return x

def parse(x):
    x = x[:,:,None]
    x = tf.tile(x, (1,1,3))    
    x = tf.image.resize(x, (SIZE, SIZE))
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def parseC(x):
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def make_dataset(X, Y, f):
    x = tf.data.Dataset.from_tensor_slices(X)
    y = tf.data.Dataset.from_tensor_slices(Y)
    x = x.map(f)
    xy = tf.data.Dataset.zip((x, y))
    xy = xy.shuffle(BUFFER_SIZE)
    return xy

def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    return xpriv, xpub

def load_cifar_100():
    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    xpriv = make_dataset(x_train, y_train, parseC)
    xpub = make_dataset(x_test, y_test, parseC)
    return xpriv, xpub

def load_cifar():
    cifar = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    xpriv = make_dataset(x_train, y_train, parseC)
    xpub = make_dataset(x_test, y_test, parseC)
    return xpriv, xpub

def load_fashion_mnist():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    
    return xpriv, xpub

def parseCelebA(x):
    x = tf.image.resize(x, (32, 32))
    # x = x[:,:,[2,1,0]]
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def get_celeba_data():
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    print('starting loading data')
    t = time.time()
    for i, line in enumerate(open('celebA/list_attr_celeba.txt', 'r')):
        image_name = line.split(' ')[0]
        label1 = int(line.split(' ')[22])
        label2 = int(line.split(' ')[32])
        label3 = int(line.split(' ')[37])
        
        if i < 5000:
            if label1 == -1 and label2 == -1 and label3 == -1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([0], dtype = np.int32))
            if label1 == -1 and label2 == -1 and label3 == 1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([1], dtype = np.int32))
            if label1 == -1 and label2 == 1 and label3 == -1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([2], dtype = np.int32))
            if label1 == 1 and label2 == -1 and label3 == -1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([3], dtype = np.int32))
            if label1 == 1 and label2 == 1 and label3 == -1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([4], dtype = np.int32))
            if label1 == 1 and label2 == -1 and label3 == 1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([5], dtype = np.int32))
            if label1 == -1 and label2 == 1 and label3 == 1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([6], dtype = np.int32))
            if label1 == 1 and label2 == 1 and label3 == 1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([7], dtype = np.int32))
        elif i < 15000:
            if label1 == -1 and label2 == -1 and label3 == -1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([0], dtype = np.int32))
            if label1 == -1 and label2 == -1 and label3 == 1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([1], dtype = np.int32))
            if label1 == -1 and label2 == 1 and label3 == -1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([2], dtype = np.int32))
            if label1 == 1 and label2 == -1 and label3 == -1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([3], dtype = np.int32))
            if label1 == 1 and label2 == 1 and label3 == -1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([4], dtype = np.int32))
            if label1 == 1 and label2 == -1 and label3 == 1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([5], dtype = np.int32))
            if label1 == -1 and label2 == 1 and label3 == 1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([6], dtype = np.int32))
            if label1 == 1 and label2 == 1 and label3 == 1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([7], dtype = np.int32))
    print('finished loading data, in {} seconds'.format(time.time() - t))
    print(len(train_data))
    print(len(test_data))
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    return train_data, np.array(train_labels), test_data, np.array(test_labels)

def get_celeba_data1():
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    print('starting loading data')
    t = time.time()
    for i, line in enumerate(open('celebA/list_attr_celeba.txt', 'r')):
        image_name = line.split(' ')[0]
        label1 = int(line.split(' ')[22])
        label2 = int(line.split(' ')[32])
        label3 = int(line.split(' ')[37])
        
        if i < 5000:
            if label1 == -1 and label2 == -1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([0], dtype = np.int32))
            if label1 == 1 and label2 == -1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([1], dtype = np.int32))
            if label1 == -1 and label2 == 1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([2], dtype = np.int32))
            if label1 == 1 and label2 == 1:
                test_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                test_labels.append(np.array([3], dtype = np.int32))
        elif i < 15000:
            if label1 == -1 and label2 == -1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([0], dtype = np.int32))
            if label1 == 1 and label2 == -1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([1], dtype = np.int32))
            if label1 == -1 and label2 == 1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([2], dtype = np.int32))
            if label1 == 1 and label2 == 1:
                train_data.append(cv2.imread('celebA/img_align_celeba/'+ image_name)) 
                train_labels.append(np.array([3], dtype = np.int32))
    print('finished loading data, in {} seconds'.format(time.time() - t))
    print(len(train_data))
    print(len(test_data))
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    return train_data, np.array(train_labels), test_data, np.array(test_labels)

# load CelebA with 8 categories
def load_celeba():
    x_train, y_train, x_test, y_test = get_celeba_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    xpriv = make_dataset(x_train, y_train, parseCelebA)
    xpub = make_dataset(x_test, y_test, parseCelebA)
    return xpriv, xpub

# load CelebA with 4 categories
def load_celeba1():
    x_train, y_train, x_test, y_test = get_celeba_data1()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    xpriv = make_dataset(x_train, y_train, parseCelebA)
    xpub = make_dataset(x_test, y_test, parseCelebA)
    return xpriv, xpub

def plot(X, label='', norm=True):
    n = len(X)
    X = (X+1) / 2 
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    for i in range(n):
        ax[i].imshow(X[i]);  
        ax[i].set(xticks=[], yticks=[], title=label)
