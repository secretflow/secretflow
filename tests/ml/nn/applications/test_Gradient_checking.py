import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import secretflow.ml.nn.applications.util as util
from secretflow.ml.nn.applications.FSHA import FSHA_ori
from secretflow.ml.nn.applications.SL import SL_ori
import secretflow.ml.nn.applications.datasets as datasets
from secretflow.ml.nn.applications.datasets import *

class test_Gradient_checking():
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
        sl = SL_ori(xpriv, xpub, 10, id_setup-1, batch_size, hparams)
        iterations = 10000
        log_frequency = 1

        fsha_model_path = 'fsha_mnist'
        sl_model_path = 'sl_mnist'
        # FSHA attack training
        LOGs_fsha, dif_category_fsha, same_category_fsha, dif_category_mean_fsha, same_category_mean_fsha, dif_variance_fsha, same_variance_fsha, gradient_fsha = fsha(iterations, fsha_model_path, 4096, verbose=True, progress_bar=False, log_frequency=log_frequency)
        # normal SplitNN training
        LOGs_sl, dif_category_sl, same_category_sl, dif_category_mean_sl, same_category_mean_sl, dif_variance_sl, same_variance_sl, gradient_sl = sl(iterations, sl_model_path, 4096, verbose=True, progress_bar=False, log_frequency=log_frequency)

        # Calculate detection score and draw pictures
        util.detection_score(sl_model_path, fsha_model_path)

