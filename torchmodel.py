import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmodule import *
from torchloss import getLossFunction


class InitLoss(nn.Module):
    def __init__(self):
        super(InitLoss, self).__init__()
        self.criterion = getLossFunction('fwrse')()
        self.metrics = getLossFunction('frse')()

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
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
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
        self.regressor = ParamRegressor()
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
        self.metrics0 = getLossFunction('frse')()
        self.criterion1 = getLossFunction('bce')()
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
        self.encoder_block1 = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False))

        # self.encoder_block2 = nn.Sequential(
        #     PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True),
        #     PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False))
        self.encoder_block2_1 = PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True)

        self.attention_branch = AttentionModel(num_features_in=feature_size * 4)

        self.encoder_block2_2 = PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False)

        self.encoder_block3 = nn.Sequential(
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1, with_conv_shortcut=False),
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2, with_conv_shortcut=True),
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False))
        self.decoder = nn.Sequential(
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
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Sigmoid()))
        self.loss = AttentionLoss()

    def forward(self, inpt, gt_posmap, gt_attention):
        x = self.layer0(inpt)
        x = self.encoder_block1(x)
        x = self.encoder_block2_1(x)
        attention = self.attention_branch(x)
        attention_features = torch.stack([x[i] * torch.exp(attention[i]) for i in range(len(x))], dim=0)

        f = self.encoder_block2_2(attention_features)
        f = self.encoder_block3(f)
        posmap = self.decoder(f)
        loss, metrics_posmap, metrics_attention = self.loss(posmap, attention, gt_posmap, gt_attention)
        return loss, metrics_posmap, metrics_attention, posmap


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

        self.model = InitPRN()

        if self.gpu_num > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.visible_devices)
        self.model.to(self.device)
        # model.cuda()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0002)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

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

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def loadWeights(self, model_path):
        if self.gpu_num > 1:
            # map_location = lambda storage, loc: storage
            self.model.module.load_state_dict(torch.load(model_path))  # , map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
