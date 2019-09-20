import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Conv2DTranspose, regularizers, Flatten
from keras.layers import Add, GlobalMaxPooling2D
from keras.layers.core import Lambda
from keras.layers import add, Flatten, Multiply
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, Callback, History
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from loss import getErrorFunction, getLossFunction

from module import DistillationModule, Conv2d_AC_BN, ResBlock, \
    CbamResBlock, Conv2d_BN_AC, PRNResBlock, Conv2d_Transpose_BN_AC, RPFOModule, DecoderModule


class RZYNet:
    def __init__(self,
                 gpu_num=1,
                 loss_function='frse',
                 optimizer=Adam(lr=1e-4),
                 ):
        self.gpu_num = gpu_num
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.paral_model = None
        self.model = None

    def loadWeights(self, model_weights_path):
        self.model.load_weights(model_weights_path)

    def compileModel(self):
        model = self.model
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

    def compileOffsetModel(self):

        # loss={'landmark_out': custom_mse_lm, 'visibility_out': custom_mse_lm, 'pose_out': custom_mse_pose,
        # 'fnf_out': 'categorical_crossentropy', 'gender_out': 'categorical_crossentropy'},
        model = self.model
        if self.gpu_num > 1:
            para_model = multi_gpu_model(model, gpus=self.gpu_num)
            para_model.compile(loss={'Posmap': getLossFunction(self.loss_function, rate=0.01), 'Offset': getLossFunction(self.loss_function, rate=1),
                                     'ParamR': getLossFunction('mae', rate=0.1), 'ParamT': getLossFunction('mae', rate=0.1),
                                     'ParamS': getLossFunction('mae', rate=0.1)},
                               optimizer=self.optimizer,
                               metrics={'Posmap': getLossFunction('frse'), 'Offset': getLossFunction('frse'),
                                        'ParamR': 'mae', 'ParamT': 'mae', 'ParamS': 'mae'})
            para_model.summary()
            self.paral_model = para_model
        else:
            model.compile(loss={'Posmap': getLossFunction(self.loss_function, rate=0), 'Offset': getLossFunction(self.loss_function, rate=5),
                                'ParamR': getLossFunction('mae', rate=1), 'ParamT': getLossFunction('mae', rate=1), 'ParamS': getLossFunction('mae', rate=1)},
                          optimizer=self.optimizer,
                          metrics={'Posmap': getLossFunction('frse'), 'Offset': getLossFunction('frse'),
                                   'ParamR': 'mae', 'ParamT': 'mae', 'ParamS': 'mae'})
            model.summary()
            self.model = model

    def buildPRNet(self):
        feature_size = 16
        inpt = Input(shape=(256, 256, 3,))
        x = Conv2d_AC_BN(inpt, filters=feature_size, kernel_size=4, strides=(1, 1), padding='same')  # 256 256 16
        # x = Conv2D(filter=size, kernel_size=, padding='same', strides=(1,1), activation='relu', name='Conv0')(inpt)
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 64 64 64
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 64 64 64
        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 8 8 512

        x = DecoderModule(x, feature_size=feature_size)
        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='same')(x)  # 256 256 3
        model = Model(inputs=inpt, outputs=x)
        self.model = model
        self.compileModel()

    def buildAttentionPRNet(self):
        feature_size = 16
        inpt = Input(shape=(256, 256, 3,))
        x = Conv2d_AC_BN(inpt, filters=feature_size, kernel_size=4, strides=(1, 1), padding='same')  # 256 256 16
        # x = Conv2D(filter=size, kernel_size=, padding='same', strides=(1,1), activation='relu', name='Conv0')(inpt)
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 64 64 64
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 64 64 64

        x = DistillationModule(x, nb_filter=feature_size * 4)

        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 8 8 512

        x = DecoderModule(x, feature_size=feature_size)
        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='same')(x)  # 256 256 3

        model = Model(inputs=inpt, outputs=x)
        self.model = model
        self.compileModel()

    def buildInitPRNet(self):
        feature_size = 16
        inpt = Input(shape=(256, 256, 3,))
        x = Conv2d_BN_AC(inpt, filters=feature_size, kernel_size=4, strides=(1, 1), padding='same')  # 256 256 16
        # x = Conv2D(filter=size, kernel_size=, padding='same', strides=(1,1), activation='relu', name='Conv0')(inpt)
        x = PRNResBlock(x, filters=feature_size * 2, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 128 128 32
        x = PRNResBlock(x, filters=feature_size * 2, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 128 128 32
        x = PRNResBlock(x, filters=feature_size * 4, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 64 64 64
        x = PRNResBlock(x, filters=feature_size * 4, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 64 64 64
        x = PRNResBlock(x, filters=feature_size * 8, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 32 32 128
        x = PRNResBlock(x, filters=feature_size * 8, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 32 32 128
        x = PRNResBlock(x, filters=feature_size * 16, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 16 16 256
        x = PRNResBlock(x, filters=feature_size * 16, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 16 16 256
        x = PRNResBlock(x, filters=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        x = PRNResBlock(x, filters=feature_size * 32, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 8 8 512

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
        x = Conv2d_Transpose_BN_AC(x, filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='same')  # 256 256 3

        model = Model(inputs=inpt, outputs=x)
        self.model = model
        self.compileModel()

    def buildCbamPRNet(self):
        feature_size = 16
        inpt = Input(shape=(256, 256, 3,))
        x = Conv2D(feature_size, kernel_size=4, strides=(1, 1), padding='same')(inpt)  # 256 256 16
        # x = Conv2D(filter=size, kernel_size=, padding='same', strides=(1,1), activation='relu', name='Conv0')(inpt)
        x = CbamResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 128 128 32
        x = CbamResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 128 128 32
        x = CbamResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 64 64 64
        x = CbamResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 64 64 64

        x = CbamResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 32 32 128
        x = CbamResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 32 32 128
        x = CbamResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 16 16 256
        x = CbamResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 16 16 256
        x = CbamResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        x = CbamResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 8 8 512

        x = DecoderModule(x)
        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='same')(x)  # 256 256 3

        model = Model(inputs=inpt, outputs=x)
        self.model = model
        self.compileModel()

    def buildOffsetPRNet(self):
        feature_size = 16
        inpt = Input(shape=(256, 256, 3,))
        x = Conv2d_BN_AC(inpt, filters=feature_size, kernel_size=4, strides=(1, 1), padding='same')  # 256 256 16
        # x = Conv2D(filter=size, kernel_size=, padding='same', strides=(1,1), activation='relu', name='Conv0')(inpt)
        x = PRNResBlock(x, filters=feature_size * 2, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 128 128 32
        x = PRNResBlock(x, filters=feature_size * 2, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 128 128 32
        x = PRNResBlock(x, filters=feature_size * 4, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 64 64 64
        x = PRNResBlock(x, filters=feature_size * 4, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 64 64 64
        x = PRNResBlock(x, filters=feature_size * 8, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 32 32 128
        x = PRNResBlock(x, filters=feature_size * 8, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 32 32 128
        x = PRNResBlock(x, filters=feature_size * 16, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 16 16 256
        x = PRNResBlock(x, filters=feature_size * 16, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 16 16 256
        x = PRNResBlock(x, filters=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        fs = PRNResBlock(x, filters=feature_size * 32, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 8 8 512

        d = Flatten()(fs)
        d = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(d)
        d = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(d)
        r = Dense(3, activation='tanh', name='ParamR')(d)
        t = Dense(3, activation='tanh', name='ParamT')(d)
        s = Dense(1, activation='sigmoid', name='ParamS')(d)

        x = DecoderModule(fs, feature_size=feature_size)  # 256 256 3
        offset = Conv2d_Transpose_BN_AC(x, filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='same', name='Offset')  # 256 256 3

        pos = Lambda(RPFOModule, output_shape=(256, 256, 3), name='Posmap')([offset, r, t, s])
        # keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)

        model = Model(inputs=inpt, outputs=[pos, offset, r, t, s])
        self.model = model
        self.compileOffsetModel()
