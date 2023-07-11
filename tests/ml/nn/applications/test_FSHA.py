import os
import tensorflow as tf
from secretflow.ml.nn.applications.FSHA import FSHA_ori
import secretflow.ml.nn.applications.util as util
import secretflow.ml.nn.applications.datasets as datasets
from secretflow.ml.nn.applications.datasets import *

class test_FSHA_DP():
    def __init__(self, dataset = 'cifar10', batch_size = 64, id_setup = 4, hparams = {
            'WGAN' : True,
            'gradient_penalty' : 500.,
            'style_loss' : None,
            'lr_f' :  0.00001,
            'lr_tilde' : 0.00001,
            'lr_D' : 0.0001,
        }):
        self.dataset = dataset
        self.batch_size = batch_size
        self.id_setup = id_setup
        self.hparams = hparams
        # dataset = 'cifar10' ## 'mnist' 

    def run(self):
        #load cifar10 dataset
        if self.dataset == 'mnist':
            xpriv, xpub = load_mnist()
        elif self.dataset == 'cifar10':
            xpriv, xpub = load_cifar()

        # hparams
        batch_size = self.batch_size
        id_setup = self.id_setup
        hparams = self.hparams

        fsha = FSHA_ori(xpriv, xpub, 10, id_setup-1, batch_size, hparams)
        iterations = 10000
        log_frequency = 10
        fsha_model_path = 'fsha_mnist'
        LOGs_fsha, dif_category_fsha, same_category_fsha, dif_category_mean_fsha, same_category_mean_fsha, dif_variance_fsha, same_variance_fsha, gradient_fsha = fsha(iterations, fsha_model_path, 4096, verbose=True, progress_bar=False, log_frequency=log_frequency)

        n = 20
        X = getImagesDS(xpriv, n)
        X_recoveredo, control = fsha.attack(X)

        def plot(X):
            n = len(X)
            X = (X+1)/2
            fig, ax = plt.subplots(1, n, figsize=(n*3,3))
            plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=-.05)
            for i in range(n):
                ax[i].imshow((X[i]), cmap='inferno');  
                ax[i].set(xticks=[], yticks=[])
                ax[i].set_aspect('equal')
                
            return fig


        fig = plot(X)
        fig = plot(X_recoveredo)