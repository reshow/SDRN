import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmodule import Conv2d_BN_AC, PRNResBlock


class InitPRN(nn.Module):
    def __init__(self):
        super(InitPRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),
        )
        self.decoder=nn.Sequential(

        )

    def forward(self, inpt):
        x = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1)(inpt)

        x = PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True)(x)  # 128 128 32
        x = PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False)(x)  # 128 128 32
        x = PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True)(x)  # 64 64 64
        x = PRNResBlock(x, filters=feature_size * 4, kernel_size=4, strides=(1, 1),
                        with_conv_shortcut=False)  # 64 64 64
        x = PRNResBlock(x, filters=feature_size * 8, kernel_size=4, strides=(2, 2),
                        with_conv_shortcut=True)  # 32 32 128
        x = PRNResBlock(x, filters=feature_size * 8, kernel_size=4, strides=(1, 1),
                        with_conv_shortcut=False)  # 32 32 128
        x = PRNResBlock(x, filters=feature_size * 16, kernel_size=4, strides=(2, 2),
                        with_conv_shortcut=True)  # 16 16 256
        x = PRNResBlock(x, filters=feature_size * 16, kernel_size=4, strides=(1, 1),
                        with_conv_shortcut=False)  # 16 16 256
        x = PRNResBlock(x, filters=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        x = PRNResBlock(x, filters=feature_size * 32, kernel_size=4, strides=(1, 1),
                        with_conv_shortcut=False)  # 8 8 512

        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 32, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 8 8 512
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 16, kernel_size=4, strides=(2, 2), activation='relu',
                                   padding='same')  # 16 16 256
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 16 16 256
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 16 16 256
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 8, kernel_size=4, strides=(2, 2), activation='relu',
                                   padding='same')  # 32 32 128
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 32 32 128
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 32 32 128
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 4, kernel_size=4, strides=(2, 2), activation='relu',
                                   padding='same')  # 64 64 64
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 64 64 64
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 64 64 64
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 2, kernel_size=4, strides=(2, 2), activation='relu',
                                   padding='same')  # 128 128 32
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size * 2, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 128 128 32
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size, kernel_size=4, strides=(2, 2), activation='relu',
                                   padding='same')  # 256 256 16
        x = Conv2d_Transpose_BN_AC(x, filters=feature_size, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 256 256 16
        x = Conv2d_Transpose_BN_AC(x, filters=3, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 256 256 3
        x = Conv2d_Transpose_BN_AC(x, filters=3, kernel_size=4, strides=(1, 1), activation='relu',
                                   padding='same')  # 256 256 3
        x = Conv2d_Transpose_BN_AC(x, filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid',
                                   padding='same')  # 256 256 3


class TorchNet:

    def __init__(self,
                 gpu_num=1,
                 loss_function='frse',
                 optimizer='adam',
                 learning_rate=1e-4
                 ):
        self.gpu_num = gpu_num
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.paral_model = None
        self.model = None
        self.learning_rate = learning_rate

    def buildInitPRN(self):
        feature_size = 16
