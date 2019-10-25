import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmodule import *
from loss import getLossFunction


class InitLoss(nn.Module):
    def __init__(self):
        super(InitLoss, self).__init__()
        self.criterion = getLossFunction('fwrse')()
        self.metrics = getLossFunction('nme')()

    def forward(self, posmap, gt_posmap):
        loss_posmap = self.criterion(gt_posmap, posmap)
        metrics_posmap = self.metrics(gt_posmap, posmap)
        return loss_posmap, metrics_posmap


class InitPRN(nn.Module):
    def __init__(self):
        super(InitPRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)
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
            # PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),
            # PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),

        )
        self.decoder = nn.Sequential(
            # output_padding = stride-1
            # padding=(kernelsize-1)//2
            # ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Sigmoid())
        )
        self.loss = InitLoss()

    def forward(self, inpt, gt):
        x = self.layer0(inpt)
        x = self.encoder(x)
        x = self.decoder(x)
        loss, metrics = self.loss(x, gt)
        return loss, metrics, x


class InitPRN2(nn.Module):
    def __init__(self):
        super(InitPRN2, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)  # 256 x 256 x 16
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2, with_conv_shortcut=True),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1, with_conv_shortcut=False),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2, with_conv_shortcut=True),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1, with_conv_shortcut=False),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2, with_conv_shortcut=True),  # 8 x 8 x 512
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),  # 8 x 8 x 512
        )
        self.decoder = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh())
        )
        self.loss = InitLoss()

    def forward(self, inpt, gt):
        x = self.layer0(inpt)
        x = self.encoder(x)
        x = self.decoder(x)
        loss, metrics = self.loss(x, gt)
        return loss, metrics, x


class OffsetLoss(nn.Module):
    def __init__(self):
        super(OffsetLoss, self).__init__()
        self.criterion0 = getLossFunction('fwrse')(0)
        self.criterion1 = getLossFunction('fwrse')(0.5)
        self.criterion2 = getLossFunction('mae')(0.5)
        self.criterion3 = getLossFunction('mae')(1)
        self.criterion4 = getLossFunction('mae')(0.25)
        self.metrics0 = getLossFunction('frse')(1.)
        self.metrics1 = getLossFunction('frse')(1.)
        self.metrics2 = getLossFunction('mae')(1.)
        self.metrics3 = getLossFunction('mae')(1.)
        self.metrics4 = getLossFunction('mae')(1.)

    def forward(self, posmap, offset, r, t, s,
                gt_posmap, gt_offset, gt_r, gt_t, gt_s):
        loss_posmap = self.criterion0(gt_posmap, posmap)
        loss_offset = self.criterion1(gt_offset, offset)
        loss_r = self.criterion2(gt_r, r)
        loss_t = self.criterion3(gt_t, t)
        loss_s = self.criterion4(gt_s, s)
        loss = loss_posmap + loss_offset + loss_r + loss_t + loss_s

        metrics_posmap = self.metrics0(gt_posmap, posmap)
        metrics_offset = self.metrics1(gt_offset, offset)
        metrics_r = self.metrics2(gt_r, r)
        metrics_t = self.metrics3(gt_t, t)
        metrics_s = self.metrics4(gt_s, s)
        return loss, metrics_posmap, metrics_offset, metrics_r, metrics_t, metrics_s


class OffsetPRN(nn.Module):
    def __init__(self):
        super(OffsetPRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=3, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1, with_conv_shortcut=False),
        )
        self.regressor = RTSRegressor()
        self.decoder = nn.Sequential(
            # output_padding = stride-1
            # padding=(kernelsize-1)//2
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=3, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=3, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=3, stride=1, activation=nn.Sigmoid()))
        self.rebuilder = RPFOModule()
        self.loss = OffsetLoss()

    def forward(self, inpt, gt_posmap, gt_offset, gt_r, gt_t, gt_s):
        x = self.layer0(inpt)
        x = self.encoder(x)

        r, t, s = self.regressor(x)
        offset = self.decoder(x)
        offset = offset * 2 - 1
        # posmap = self.rebuilder(offset, r, t, s)
        posmap = self.rebuilder(offset, gt_r, gt_t, torch.unsqueeze(gt_s, 1))

        loss, metrics_posmap, metrics_offset, metrics_r, metrics_t, metrics_s = self.loss(posmap, offset, r, t, s,
                                                                                          gt_posmap, gt_offset, gt_r, gt_t, gt_s)
        return loss, metrics_posmap, metrics_offset, metrics_r, metrics_t, metrics_s, posmap


class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.criterion0 = getLossFunction('fwrse')()
        self.metrics0 = getLossFunction('nme')()
        self.criterion1 = getLossFunction('bce')(0.1)
        self.metrics1 = getLossFunction('mae')()

    def forward(self, posmap, mask, gt_posmap, gt_mask):
        loss_posmap = self.criterion0(gt_posmap, posmap)
        metrics_posmap = self.metrics0(gt_posmap, posmap)
        loss_mask = self.criterion1(gt_mask, mask)
        metrics_attention = self.metrics1(gt_mask, mask)
        loss = loss_posmap + loss_mask
        return loss, metrics_posmap, metrics_attention


class AttentionPRN(nn.Module):
    def __init__(self):
        super(AttentionPRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)

        self.block1 = PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True)  # 128 x 128 x 32
        self.block2 = PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1,
                                  with_conv_shortcut=False)  # 128 x 128 x 32
        self.block3 = PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2,
                                  with_conv_shortcut=True)  # 64 x 64 x 64
        self.block4 = PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1,
                                  with_conv_shortcut=False)  # 64 x 64 x 64
        self.block5 = PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2,
                                  with_conv_shortcut=True)  # 32 x 32 x 128
        self.block6 = PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1,
                                  with_conv_shortcut=False)  # 32 x 32 x 128
        self.block7 = PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2,
                                  with_conv_shortcut=True)  # 16 x 16 x 256
        self.block8 = PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1,
                                  with_conv_shortcut=False)  # 16 x 16 x 256
        self.block9 = PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2,
                                  with_conv_shortcut=True)  # 8 x 8 x 512
        self.block10 = PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1,
                                   with_conv_shortcut=False)  # 8 x 8 x 512

        self.attention_branch = AttentionModel(num_features_in=feature_size * 8)

        self.decoder = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh())
        )
        self.loss = AttentionLoss()

    def forward(self, inpt, gt_posmap, gt_attention):
        x = self.layer0(inpt)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        attention = self.attention_branch(x)

        # a=attention.squeeze().cpu().numpy()
        # import visualize
        # visualize.showImage(np.exp(a),False)

        attention_features = torch.stack([x[i] * torch.exp(attention[i]) for i in range(len(x))], dim=0)

        f = self.block7(attention_features)
        f = self.block8(f)
        f = self.block9(f)
        f = self.block10(f)
        posmap = self.decoder(f)
        loss, metrics_posmap, metrics_attention = self.loss(posmap, attention, gt_posmap, gt_attention)
        return loss, metrics_posmap, metrics_attention, posmap


class QuaternionOffsetLoss(nn.Module):
    def __init__(self):
        super(QuaternionOffsetLoss, self).__init__()
        self.criterion0 = getLossFunction('fwrse')(0)
        self.criterion1 = getLossFunction('fwrse')(1)
        self.criterion2 = getLossFunction('rmse')(3)
        self.criterion3 = getLossFunction('rmse')(3)
        self.metrics0 = getLossFunction('frse')(1.)
        self.metrics1 = getLossFunction('frse')(1.)
        self.metrics2 = getLossFunction('mae')(1.)
        self.metrics3 = getLossFunction('mae')(1.)

    def forward(self, posmap, offset, q, t2d,
                gt_posmap, gt_offset, gt_q, gt_t):
        loss_posmap = self.criterion0(gt_posmap, posmap)
        loss_offset = self.criterion1(gt_offset, offset)
        loss_q = self.criterion2(gt_q, q)
        loss_t = self.criterion3(gt_t[:, 0:2], t2d)
        loss = loss_posmap + loss_offset + loss_q + loss_t

        metrics_posmap = self.metrics0(gt_posmap, posmap)
        metrics_offset = self.metrics1(gt_offset, offset)
        metrics_q = self.metrics2(gt_q, q)
        metrics_t = self.metrics3(gt_t[:, 0:2], t2d)
        return loss, metrics_posmap, metrics_offset, metrics_q, metrics_t


class QuaternionOffsetPRN(nn.Module):
    def __init__(self):
        super(QuaternionOffsetPRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2, with_conv_shortcut=True),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1, with_conv_shortcut=False),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2, with_conv_shortcut=True),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1, with_conv_shortcut=False),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2, with_conv_shortcut=True),  # 8 x 8 x 512
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),  # 8 x 8 x 512
        )
        self.regressor = QTRegressor(filters=feature_size * 32)
        self.decoder = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh()))
        self.rebuilder = RPFQModule()
        self.loss = QuaternionOffsetLoss()

    def forward(self, inpt, gt_posmap, gt_offset, gt_q, gt_t):
        x = self.layer0(inpt)
        x = self.encoder(x)

        q, t2d = self.regressor(x)
        offset = self.decoder(x)

        # posmap = self.rebuilder(offset, r, t, s)
        t3d = torch.zeros((inpt.shape[0], 3))
        t3d = t3d.to(t2d.device)
        t3d[:, 0:2] = t2d
        t3d[:, 2] = gt_t[:, 2]
        # posmap = self.rebuilder(offset, gt_q, gt_t)
        posmap = self.rebuilder(offset, q, t3d)

        loss, metrics_posmap, metrics_offset, metrics_q, metrics_t = self.loss(posmap, offset, q, t2d,
                                                                               gt_posmap, gt_offset, gt_q, gt_t)
        return loss, metrics_posmap, metrics_offset, metrics_q, metrics_t, posmap


class SiamLoss(nn.Module):
    def __init__(self):
        super(SiamLoss, self).__init__()
        # self.criterion0 = getLossFunction('fwrse')(0)
        self.criterion1 = getLossFunction('fwrse')(0.5)
        self.criterion2 = getLossFunction('fwrse')(1)
        self.metrics0 = getLossFunction('nme')(1.)
        self.metrics1 = getLossFunction('nme')(1.)
        self.metrics2 = getLossFunction('nme')(1.)

    def forward(self, posmap, offset, kpt_posmap,
                gt_posmap, gt_offset):
        # loss_posmap = self.criterion0(gt_posmap, posmap)
        loss_offset = self.criterion1(gt_offset, offset)
        loss_kpt = self.criterion2(gt_posmap, kpt_posmap)
        loss = loss_offset + loss_kpt

        metrics_posmap = self.metrics0(gt_posmap, posmap)
        metrics_offset = self.metrics1(gt_offset, offset)
        metrics_kpt = self.metrics2(gt_posmap, kpt_posmap)
        return loss, metrics_posmap, metrics_offset, metrics_kpt


class SiamPRN(nn.Module):
    def __init__(self):
        super(SiamPRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)  # 256 x 256 x 16
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2, with_conv_shortcut=True),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1, with_conv_shortcut=False),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2, with_conv_shortcut=True),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1, with_conv_shortcut=False),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2, with_conv_shortcut=True),  # 8 x 8 x 512
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),  # 8 x 8 x 512
        )
        self.decoder = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh()))
        self.decoder_kpt = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh()))
        self.rebuilder = EstimateRebuildModule()
        self.loss = SiamLoss()

    def forward(self, inpt, gt_posmap, gt_offset, is_rebuild=False):
        x = self.layer0(inpt)
        x = self.encoder(x)
        x_new=x.detach()
        offset = self.decoder(x_new)

        kpt_posmap = self.decoder_kpt(x)

        if is_rebuild:
            posmap = self.rebuilder(offset, kpt_posmap)
        else:
            if self.training:
                posmap = gt_posmap.clone()
            else:
                posmap = self.rebuilder(offset, kpt_posmap)

        loss, metrics_posmap, metrics_offset, metrics_kpt = self.loss(posmap, offset, kpt_posmap, gt_posmap, gt_offset)
        return loss, metrics_posmap, metrics_offset, metrics_kpt, posmap


class MeanOffsetLoss(nn.Module):
    def __init__(self):
        super(MeanOffsetLoss, self).__init__()
        self.criterion0 = getLossFunction('fwrse')(0)
        self.criterion1 = getLossFunction('fwrse')(1)
        self.criterion2 = getLossFunction('rmse')(2)
        self.criterion3 = getLossFunction('rmse')(2)
        self.metrics0 = getLossFunction('frse')(1.)
        self.metrics1 = getLossFunction('frse')(1.)
        self.metrics2 = getLossFunction('mae')(1.)
        self.metrics3 = getLossFunction('mae')(1.)

    def forward(self, posmap, offset, qs, t2ds,
                gt_posmap, gt_offset, gt_q, gt_t,
                num_cluster):
        loss_posmap = self.criterion0(gt_posmap, posmap)
        loss_offset = self.criterion1(gt_offset, offset)

        loss_q = 0
        loss_t = 0
        assert (num_cluster % 2 == 0)

        for i in range(int(num_cluster / 2)):
            loss_q += self.criterion2(gt_q + i * 0.1, qs[:, :, i])
            loss_t += self.criterion3(gt_t[:, 0:2] + i * 0.1, t2ds[:, :, i])
        for i in range(int(num_cluster / 2)):
            loss_q += self.criterion2(gt_q - i * 0.1, qs[:, :, i + int(num_cluster / 2)])
            loss_t += self.criterion3(gt_t[:, 0:2] - i * 0.1, t2ds[:, :, i + int(num_cluster / 2)])

        loss = loss_posmap + loss_offset + loss_q / num_cluster + loss_t / num_cluster

        metrics_posmap = self.metrics0(gt_posmap, posmap)
        metrics_offset = self.metrics1(gt_offset, offset)

        q = torch.mean(qs, 2)
        t2d = torch.mean(t2ds, 2)

        metrics_q = self.metrics2(gt_q, q)
        metrics_t = self.metrics3(gt_t[:, 0:2], t2d)
        return loss, metrics_posmap, metrics_offset, metrics_q, metrics_t


class MeanOffsetPRN(nn.Module):
    def __init__(self):
        super(MeanOffsetPRN, self).__init__()
        self.feature_size = 16
        self.num_cluster = 10
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2, with_conv_shortcut=True),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1, with_conv_shortcut=False),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2, with_conv_shortcut=True),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1, with_conv_shortcut=False),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2, with_conv_shortcut=True),  # 8 x 8 x 512
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),  # 8 x 8 x 512
        )
        self.regressor = MeanQTRegressor(num_cluster=self.num_cluster, filters=feature_size * 32)
        self.decoder = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh()))
        self.rebuilder = RPFQModule()
        self.loss = MeanOffsetLoss()

    def forward(self, inpt, gt_posmap, gt_offset, gt_q, gt_t):
        x = self.layer0(inpt)
        x = self.encoder(x)

        qs, t2ds = self.regressor(x)
        offset = self.decoder(x)

        qs = qs.reshape(qs.shape[0], 4, self.num_cluster)
        t2ds = t2ds.reshape(qs.shape[0], 2, self.num_cluster)
        q = torch.mean(qs, 2)
        t2d = torch.mean(t2ds, 2)

        # posmap = self.rebuilder(offset, r, t, s)
        t3d = torch.zeros((inpt.shape[0], 3))
        t3d = t3d.to(t2d.device)
        t3d[:, 0:2] = t2d
        t3d[:, 2] = gt_t[:, 2]
        # posmap = self.rebuilder(offset, gt_q, gt_t)
        posmap = self.rebuilder(offset, q, t3d)

        loss, metrics_posmap, metrics_offset, metrics_q, metrics_t = self.loss(posmap, offset, qs, t2ds,
                                                                               gt_posmap, gt_offset, gt_q, gt_t,
                                                                               self.num_cluster)
        return loss, metrics_posmap, metrics_offset, metrics_q, metrics_t, posmap


class TorchNet:

    def __init__(self,
                 gpu_num=1,
                 visible_gpus='0',
                 learning_rate=1e-4
                 ):
        self.gpu_num = gpu_num
        gpus = visible_gpus.split(',')
        self.visible_devices = [int(i) for i in gpus]

        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device("cuda:" + gpus[0] if torch.cuda.is_available() else "cpu")

    def buildInitPRN(self):

        self.model = InitPRN2()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0002)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def buildOffsetPRN(self):

        self.model = OffsetPRN()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def buildAttentionPRN(self):
        self.model = AttentionPRN()
        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0002)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def buildQuaternionOffsetPRN(self):

        self.model = QuaternionOffsetPRN()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def buildSiamPRN(self):

        self.model = SiamPRN()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0002)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def buildMeanOffsetPRN(self):

        self.model = MeanOffsetPRN()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def loadWeights(self, model_path):
        if self.gpu_num > 1:
            # map_location = lambda storage, loc: storage
            self.model.module.load_state_dict(torch.load(model_path))  # , map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
            # self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
