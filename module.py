import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Conv2DTranspose, regularizers, Dot, Permute
from keras.layers import Add, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Permute, multiply
from keras.layers import add, Flatten, Multiply, Lambda, Concatenate
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, Callback, History
from keras.utils import multi_gpu_model
import keras.backend as K
from data import mean_posmap

# global variable
mean_posmap_tensor = K.variable(mean_posmap)


def Conv2d_AC_BN(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    return x


def Conv2d_BN_AC(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_regularizer=regularizers.l2(0.0002))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def Conv2d_Transpose_BN_AC(x, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', name=None):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=regularizers.l2(0.0002))(x)
    x = BatchNormalization()(x)
    if name is None:
        x = Activation(activation)(x)
    else:
        x = Activation(activation, name=name)(x)
    return x


def ResBlock(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_AC_BN(inpt, filters=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_AC_BN(x, filters=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_AC_BN(inpt, filters=nb_filter, strides=strides, kernel_size=kernel_size)
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
    x1 = Conv2d_AC_BN(inpt, filters=nb_filter, kernel_size=1, strides=(1, 1), padding='same')
    x1 = Conv2d_AC_BN(x1, filters=nb_filter, kernel_size=3, padding='same')
    x1 = Conv2d_AC_BN(x1, filters=nb_filter, kernel_size=1, padding='same')
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


def channel_attention(x, ratio):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = x._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([x, cbam_feature])


def spatial_attention(x):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = x._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(x)
    else:
        channel = x._keras_shape[-1]
        cbam_feature = x

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([x, cbam_feature])


def CbamModule(x, ratio=4):
    x = channel_attention(x, ratio)
    x = spatial_attention(x)
    return x


def CbamResBlock(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN_AC(inpt, filters=int(nb_filter / 2), kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN_AC(x, filters=int(nb_filter / 2), kernel_size=kernel_size, padding='same')
    x = Conv2d_BN_AC(x, filters=nb_filter, kernel_size=1, padding='same')
    # x = CbamModule(x)
    # if with_conv_shortcut:
    #     shortcut = Conv2d_BN_AC(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
    #     x = add([x, shortcut])
    #     x = Activation('relu')(x)
    #     return x
    # else:
    #     x = add([x, inpt])
    #     x = Activation('relu')(x)
    #     return x

    # x = Conv2D(int(nb_filter / 2), kernel_size=1, strides=(1, 1), padding='same')(inpt)
    # x = Conv2D(int(nb_filter / 2), kernel_size=kernel_size, strides=strides, padding='same')(x)
    # x = Conv2D(nb_filter, kernel_size=1, strides=(1, 1), padding='same')(x)
    x = CbamModule(x)
    if with_conv_shortcut:
        shortcut = Conv2D(nb_filter, strides=strides, kernel_size=1)(inpt)
        x = add([x, shortcut])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    else:
        x = add([x, inpt])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x


def PRNResBlock(inpt, filters, kernel_size=4, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN_AC(inpt, int(filters / 2), kernel_size=1, strides=(1, 1), padding='same')
    x = Conv2d_BN_AC(x, int(filters / 2), kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2D(filters, kernel_size=1, strides=(1, 1), padding='same')(x)
    if with_conv_shortcut:
        shortcut = Conv2D(filters, strides=strides, kernel_size=1)(inpt)
        x = add([x, shortcut])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    else:
        x = add([x, inpt])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x


def getRotateTensor(R_flatten):
    x = R_flatten[:, 0:1]
    y = R_flatten[:, 1:2]
    z = R_flatten[:, 2:3]

    rx1 = K.concatenate([x / x, x * 0, x * 0])
    rx2 = K.concatenate([x * 0, K.cos(x), K.sin(x)])
    rx3 = K.concatenate([x * 0, -K.sin(x), K.cos(x)])
    rx = K.stack([rx1, rx2, rx3], axis=1)

    ry1 = K.concatenate([K.cos(y), y * 0, -K.sin(y)])
    ry2 = K.concatenate([y * 0, y / y, y * 0])
    ry3 = K.concatenate([K.sin(y), y * 0, K.cos(y)])
    ry = K.stack([ry1, ry2, ry3], axis=1)

    rz1 = K.concatenate([K.cos(z), K.sin(z), z * 0])
    rz2 = K.concatenate([-K.sin(z), K.cos(z), z * 0])
    rz3 = K.concatenate([z * 0, z * 0, z / z])
    rz = K.stack([rz1, rz2, rz3], axis=1)

    r1 = Dot([2, 1])([rx, ry])
    r2 = Dot([2, 1])([r1, rz])
    return r2


# RestorePositionFromOffset
def RPFOModule(inpt):
    """
        offset: 256*256*3
        T: 12*1
        :return: posmap: 256*256*3
    """
    [offset, R, T, S] = inpt
    R = R * np.pi
    Sn = -S
    s = K.concatenate([S, Sn, S])
    s = K.repeat(s, 3)

    r = getRotateTensor(R)
    # 1e-4*5e2    *20=1
    r = r * s * 20.
    t = T * 280.
    pos = offset*2 + mean_posmap_tensor
    pos = Reshape((65536, 3))(pos)
    tk1 = K.repeat(t, 65536)

    pos = Dot(2)([pos, r])
    pos = pos + tk1
    pos = Reshape((256, 256, 3))(pos)
    pos = pos / 256.

    return pos

    # batch_size = T.shape[0]
    # outpt = []
    # for i in range(batch_size):
    #     r = K.reshape(R[i] * 10., (3, 3))
    #     t = T[i] * 256.
    #     pos = K.dot((offset[i] + mean_posmap_tensor), K.transpose(r))
    #     pos = pos + t


def DecoderModule(x, feature_size=16):
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 32, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 8 8 512
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 16, kernel_size=4, strides=(2, 2), activation='relu', padding='same')  # 16 16 256
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 16 16 256
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 16 16 256
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 8, kernel_size=4, strides=(2, 2), activation='relu', padding='same')  # 32 32 128
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 32 32 128
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 32 32 128
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 4, kernel_size=4, strides=(2, 2), activation='relu', padding='same')  # 64 64 64
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 64 64 64
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 64 64 64
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 2, kernel_size=4, strides=(2, 2), activation='relu', padding='same')  # 128 128 32
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 2, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 128 128 32
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size, kernel_size=4, strides=(2, 2), activation='relu', padding='same')  # 256 256 16
    x = Conv2d_Transpose_BN_AC(x, filters=feature_size, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 256 256 16
    x = Conv2d_Transpose_BN_AC(x, filters=3, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 256 256 3
    x = Conv2d_Transpose_BN_AC(x, filters=3, kernel_size=4, strides=(1, 1), activation='relu', padding='same')  # 256 256 3
    return x
