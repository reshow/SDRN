import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdata import mean_posmap


# Hout​=(Hin​−1)stride[0]−2padding[0]+kernels​ize[0]+outputp​adding[0]

class Conv2d_BN_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, padding_mode='zeros'):
        super(Conv2d_BN_AC, self).__init__()
        self.pipe = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        out = self.pipe(x)
        return out


class ConvTranspose2d_BN_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.ReLU()):
        super(ConvTranspose2d_BN_AC, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, output_padding=stride - 1)

        self.BN_AC = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            activation)

        self.crop_size = (kernel_size + 1) % 2

    def forward(self, x):
        out = self.deconv(x)
        out2 = out[:, :, self.crop_size:out.shape[2], self.crop_size:out.shape[3]].clone()
        out2 = self.BN_AC(out2)
        return out2


class PRNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_conv_shortcut=False):
        super(PRNResBlock, self).__init__()

        if kernel_size % 2 == 1:
            self.pipe = nn.Sequential(
                Conv2d_BN_AC(in_channels=in_channels, out_channels=out_channels // 2, stride=1, kernel_size=1),
                Conv2d_BN_AC(in_channels=out_channels // 2, out_channels=out_channels // 2, stride=stride,
                             kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, stride=1, kernel_size=1),

            )
        else:
            self.pipe = nn.Sequential(
                Conv2d_BN_AC(in_channels=in_channels, out_channels=out_channels // 2, stride=1, kernel_size=1),
                Conv2d_BN_AC(in_channels=out_channels // 2, out_channels=out_channels // 2, stride=stride,
                             kernel_size=kernel_size, padding=kernel_size - 1, padding_mode='circular'),
                nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, stride=1, kernel_size=1)
            )
        self.shortcut = nn.Sequential()

        if with_conv_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1),
            )
        self.BN_AC = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.pipe(x)
        out = out + self.shortcut(x)
        out = self.BN_AC(out)
        return out


# RestorePositionFromOffset

def getRotationTensor(R_flatten):
    x = R_flatten[:, 0]
    y = R_flatten[:, 1]
    z = R_flatten[:, 2]
    rx = torch.zeros((R_flatten.shape[0], 3, 3), device=R_flatten.device)
    ry = torch.zeros((R_flatten.shape[0], 3, 3), device=R_flatten.device)
    rz = torch.zeros((R_flatten.shape[0], 3, 3), device=R_flatten.device)

    rx[:, 0, 0] = 1
    rx[:, 1, 1] = torch.cos(x)
    rx[:, 1, 2] = torch.sin(x)
    rx[:, 2, 1] = torch.sin(-x)
    rx[:, 2, 2] = torch.cos(x)

    ry[:, 1, 1] = 1
    ry[:, 0, 0] = torch.cos(y)
    ry[:, 2, 0] = torch.sin(y)
    ry[:, 0, 2] = torch.sin(-y)
    ry[:, 2, 2] = torch.cos(y)

    rz[:, 2, 2] = 1
    rz[:, 1, 1] = torch.cos(z)
    rz[:, 0, 1] = torch.sin(z)
    rz[:, 1, 0] = torch.sin(-z)
    rz[:, 0, 0] = torch.cos(z)

    outr = torch.zeros((R_flatten.shape[0], 3, 3), device=R_flatten.device)
    for i in range(R_flatten.shape[0]):
        outr[i] = rx[i].mm(ry[i]).mm(rz[i])
    return outr


# torch.autograd.set_detect_anomaly(True)


class RPFOModule(nn.Module):
    def __init__(self):
        super(RPFOModule, self).__init__()
        self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_posmap.transpose((2, 0, 1))))
        self.mean_posmap_tensor.requires_grad = False
        self.T_scale = 300
        self.S_scale = 1e4 / 5e2
        self.offset_scale = 4

    def forward(self, Offset, R, T, S):
        R = R * np.pi
        Sn = -S
        s = torch.stack((S, Sn, S), 2)
        s = s.repeat(1, 3, 1)
        s = s * self.S_scale
        r = getRotationTensor(R)
        # r = R.reshape((R.shape[0], 3, 3))
        r = r * s
        r = r.permute(0, 2, 1)
        t = T * self.T_scale
        pos = Offset * self.offset_scale + self.mean_posmap_tensor
        pos = pos.permute(0, 2, 3, 1)
        pos = pos.reshape((pos.shape[0], 65536, 3))

        outpos = pos.clone()
        for i in range(pos.shape[0]):
            pos[i] = outpos[i].mm(r[i]) + t[i]

        pos = pos.reshape((pos.shape[0], 256, 256, 3))
        pos = pos.permute(0, 3, 1, 2)
        pos = pos / 256.
        return pos


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        return out


class ParamRegressor(nn.Module):
    def __init__(self, num_cluster=1, filters=512):
        super(ParamRegressor, self).__init__()
        self.pipe = nn.Sequential(
            nn.AvgPool2d(8, stride=1),
            Flatten(),
            nn.Linear(filters, filters),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Linear(filters, filters),
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )
        self.R_layer = nn.Sequential(nn.Linear(filters, num_cluster * 3), nn.Sigmoid())
        self.T_layer = nn.Sequential(nn.Linear(filters, num_cluster * 3), nn.Sigmoid())
        self.S_layer = nn.Sequential(nn.Linear(filters, num_cluster), nn.Sigmoid())

    def forward(self, x):
        x_new = x.detach()
        feat = self.pipe(x_new)
        R = self.R_layer(feat)
        R = R * 2 - 1
        T = self.T_layer(feat)
        T = T * 2 - 1
        S = self.S_layer(feat)
        return R, T, S
