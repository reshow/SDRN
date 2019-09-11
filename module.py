import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Conv2DTranspose, regularizers
from keras.layers import Add, GlobalMaxPooling2D
from keras.layers import add, Flatten, Multiply
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, Callback, History
from keras.utils import multi_gpu_model
from loss import getErrorFunction, getLossFunction
import keras.backend as K


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


def OffsetL1ActivityRegular(l=0.01, rate=0.25):
    def ActivityL1Loss(feat):
        return l * K.abs(K.mean(K.abs(feat)) - rate)

    return ActivityL1Loss


def DistillationModule(inpt, nb_filter):
    # pathA:  kernel size 1 3 1
    x1 = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=1, strides=(1, 1), padding='same')
    x1 = Conv2d_BN(x1, nb_filter=nb_filter, kernel_size=3, padding='same')
    x1 = Conv2d_BN(x1, nb_filter=nb_filter, kernel_size=1, padding='same')
    x1 = add([x1, inpt])

    # pathB: attention  kernel size 1 3 1
    x2 = Conv2D(nb_filter, kernel_size=1, padding='same', strides=(1, 1), activation=None)(inpt)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(nb_filter, kernel_size=3, padding='same', strides=(1, 1), activation=None)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(1, kernel_size=1, padding='same', strides=(1, 1), activation=None,
                activity_regularizer=regularizers.l1(1e-3))(x2)
    # activity_regularizer=OffsetL1ActivityRegular(0.1, 0.4))(x2)

    x2 = Activation('sigmoid')(x2)
    otp = Multiply()([x1, x2])

    return otp
