import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Conv2DTranspose
from keras.layers import Add, GlobalMaxPooling2D
from keras.layers import add, Flatten
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, Callback, History
from keras.utils import multi_gpu_model
from loss import getErrorFunction, getLossFunction


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def ResBlock(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


class RZYNet:
    def __init__(self,
                 gpu_num=1,
                 loss_function='frse',
                 optimizer='adam',
                 ):
        self.gpu_num = gpu_num
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.paral_model = None
        self.model = None

    def loadWeights(self, model_weights_path):
        self.model.load_weights(model_weights_path)

    def buildPRNet(self):
        feature_size = 16
        inpt = Input(shape=(256, 256, 3,))
        x = Conv2d_BN(inpt, nb_filter=feature_size, kernel_size=4, strides=(1, 1), padding='same')  # 256 256 16
        # x = Conv2D(filter=size, kernel_size=, padding='same', strides=(1,1), activation='relu', name='Conv0')(inpt)
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(2, 2),
                     with_conv_shortcut=True)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(1, 1),
                     with_conv_shortcut=False)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 64 64 64
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 64 64 64
        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(1, 1),
                     with_conv_shortcut=False)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(2, 2),
                     with_conv_shortcut=True)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(1, 1),
                     with_conv_shortcut=False)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 8 8 512

        x = Conv2DTranspose(filters=feature_size * 32, kernel_size=4, strides=(1, 1), activation='relu',
                            padding='same')(
            x)  # 8 8 512
        x = Conv2DTranspose(filters=feature_size * 16, kernel_size=4, strides=(2, 2), activation='relu',
                            padding='same')(
            x)  # 16 16 256
        x = Conv2DTranspose(filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu',
                            padding='same')(
            x)  # 16 16 256
        x = Conv2DTranspose(filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu',
                            padding='same')(
            x)  # 16 16 256
        x = Conv2DTranspose(filters=feature_size * 8, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 32 32 128
        x = Conv2DTranspose(filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 32 32 128
        x = Conv2DTranspose(filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 32 32 128
        x = Conv2DTranspose(filters=feature_size * 4, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 64 64 64
        x = Conv2DTranspose(filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 64 64 64
        x = Conv2DTranspose(filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 64 64 64

        x = Conv2DTranspose(filters=feature_size * 2, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 128 128 32
        x = Conv2DTranspose(filters=feature_size * 2, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 128 128 32
        x = Conv2DTranspose(filters=feature_size, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 256 256 16
        x = Conv2DTranspose(filters=feature_size, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 256 256 16

        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(x)  # 256 256 3
        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(x)  # 256 256 3
        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='same')(
            x)  # 256 256 3

        model = Model(inputs=inpt, outputs=x)
        self.model = model

        if self.gpu_num > 1:
            para_model = multi_gpu_model(model, gpus=self.gpu_num)
            para_model.compile(loss=getLossFunction(self.loss_function),
                               optimizer=self.optimizer,
                               metrics=[getLossFunction('frse'), 'mae'])
            para_model.summary()
            self.paral_model = para_model
        else:
            model.compile(loss=getLossFunction(self.loss_function),
                          optimizer=self.optimizer,
                          metrics=[getLossFunction('frse'), 'mae'])
            model.summary()
            self.model = model
