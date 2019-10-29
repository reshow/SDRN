import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import mean_posmap, uv_kpt
from skimage import io, transform


# Hout​=(Hin​−1)stride[0]−2padding[0]+kernels​ize[0]+outputp​adding[0]

class Conv2d_BN_AC2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, padding_mode='zeros'):
        super(Conv2d_BN_AC2, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5)
        self.ac = nn.ReLU(inplace=True)

    def forward(self, x):
        print('\n', self.bn.running_mean[0].cpu().detach().numpy(), self.bn.running_var[0].cpu().detach().numpy(), self.bn.running_mean.shape)
        print(self.bn.weight[0].cpu().detach().numpy(), self.bn.bias[0].cpu().detach().numpy(), self.bn.weight.shape)
        print('mean:', x[0, 0].cpu().detach().numpy().mean(), ' var', x[0, 0].cpu().detach().numpy().var())
        x = self.conv(x)
        print('minibatch_mean', x[:, 0].cpu().detach().numpy().mean(), ' minibatch_var', x[:, 0].cpu().detach().numpy().var())
        print('mean:', x[0, 0].cpu().detach().numpy().mean(), ' var', x[0, 0].cpu().detach().numpy().var())

        x = self.bn(x)
        print('mean', x[0, 0].cpu().detach().numpy().mean(), '  var', x[0, 0].cpu().detach().numpy().var())
        out = self.ac(x)
        return out


class Conv2d_BN_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, padding_mode='zeros'):
        super(Conv2d_BN_AC, self).__init__()
        self.pipe = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.pipe(x)
        return out


class ConvTranspose2d_BN_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True)):
        super(ConvTranspose2d_BN_AC, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, output_padding=stride - 1, bias=False)

        self.BN_AC = nn.Sequential(
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5),
            activation)

        self.crop_size = (kernel_size + 1) % 2

    def forward(self, x):
        out = self.deconv(x)
        out2 = out[:, :, self.crop_size:out.shape[2], self.crop_size:out.shape[3]].clone()
        out2 = self.BN_AC(out2)
        return out2


class ConvTranspose2d_BN_AC2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, activation=nn.ReLU(inplace=True)):
        super(ConvTranspose2d_BN_AC2, self).__init__()
        if stride % 2 == 0:
            self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False)
        else:
            self.deconv = nn.Sequential(nn.ConstantPad2d((2, 1, 2, 1), 0),
                                        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                           kernel_size=kernel_size, stride=stride, padding=3, bias=False))

        self.BN_AC = nn.Sequential(
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5),
            activation)

    def forward(self, x):
        out = self.deconv(x)
        out2 = self.BN_AC(out)
        return out2


class PRNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_conv_shortcut=False, activation=nn.ReLU(inplace=True)):
        super(PRNResBlock, self).__init__()

        if kernel_size % 2 == 1:
            self.pipe = nn.Sequential(
                Conv2d_BN_AC(in_channels=in_channels, out_channels=out_channels // 2, stride=1, kernel_size=1),
                Conv2d_BN_AC(in_channels=out_channels // 2, out_channels=out_channels // 2, stride=stride,
                             kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, stride=1, kernel_size=1, bias=False),

            )
        else:
            self.pipe = nn.Sequential(
                Conv2d_BN_AC(in_channels=in_channels, out_channels=out_channels // 2, stride=1, kernel_size=1),
                Conv2d_BN_AC(in_channels=out_channels // 2, out_channels=out_channels // 2, stride=stride,
                             kernel_size=kernel_size, padding=kernel_size - 1, padding_mode='circular'),
                nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, stride=1, kernel_size=1, bias=False)
            )
        self.shortcut = nn.Sequential()

        if with_conv_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1, bias=False),
            )
        self.BN_AC = nn.Sequential(
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.5),
            activation
        )

    def forward(self, x):
        out = self.pipe(x)
        s = self.shortcut(x)
        assert (s.shape == out.shape)
        out = out + s
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

# rebuild posmap from output
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
        pos = pos / 280.
        return pos


def quaternion2RotationTensor(Q):
    q0 = Q[:, 0]
    q1 = Q[:, 1]
    q2 = Q[:, 2]
    q3 = Q[:, 3]
    outr = torch.zeros((Q.shape[0], 3, 3), device=Q.device)
    for i in range(Q.shape[0]):
        outr[i, 0, 0] = q0[i] ** 2 + q1[i] ** 2 - q2[i] ** 2 - q3[i] ** 2
        outr[i, 0, 1] = 2 * (q1[i] * q2[i] + q0[i] * q3[i])
        outr[i, 0, 2] = 2 * (q1[i] * q3[i] - q0[i] * q2[i])
        outr[i, 1, 0] = 2 * (q1[i] * q2[i] - q0[i] * q3[i])
        outr[i, 1, 1] = q0[i] ** 2 - q1[i] ** 2 + q2[i] ** 2 - q3[i] ** 2
        outr[i, 1, 2] = 2 * (q0[i] * q1[i] + q2[i] * q3[i])
        outr[i, 2, 0] = 2 * (q0[i] * q2[i] + q1[i] * q3[i])
        outr[i, 2, 1] = 2 * (q2[i] * q3[i] - q0[i] * q1[i])
        outr[i, 2, 2] = q0[i] ** 2 - q1[i] ** 2 - q2[i] ** 2 + q3[i] ** 2
    return outr


class RPFQModule(nn.Module):
    def __init__(self):
        super(RPFQModule, self).__init__()
        self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_posmap.transpose((2, 0, 1))))
        self.mean_posmap_tensor.requires_grad = False
        self.T_scale = 300
        self.Q_scale = 5e-2
        self.S_scale = 1e4
        self.offset_scale = 4
        revert_opetator = np.array([[1., -1., 1.], [1., -1., 1.], [1., -1., 1.]]).astype(np.float32)
        self.revert_operator = nn.Parameter(torch.from_numpy(revert_opetator))
        self.revert_operator.requires_grad = False

    def forward(self, Offset, Q, T):
        s = self.revert_operator.unsqueeze(0)
        s = s.repeat(Offset.shape[0], 1, 1)
        s = s * self.S_scale
        r = quaternion2RotationTensor(Q * self.Q_scale)
        r = r * s
        r = r.permute(0, 2, 1)  # r.Transpose
        t = T * self.T_scale
        pos = Offset * self.offset_scale + self.mean_posmap_tensor
        pos = pos.permute(0, 2, 3, 1)
        pos = pos.reshape((pos.shape[0], 65536, 3))
        outpos = pos.clone()
        for i in range(pos.shape[0]):
            pos[i] = outpos[i].mm(r[i]) + t[i]
        pos = pos.reshape((pos.shape[0], 256, 256, 3))
        pos = pos.permute(0, 3, 1, 2)
        pos = pos / 280.
        return pos


class EstimateRebuildModule(nn.Module):
    def __init__(self, is_visible=True):
        super(EstimateRebuildModule, self).__init__()
        self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_posmap.transpose((2, 0, 1))))
        self.mean_posmap_tensor.requires_grad = False

        self.S_scale = 1e4
        self.offset_scale = 6
        revert_opetator = np.array([[1., -1., 1.], [1., -1., 1.], [1., -1., 1.]]).astype(np.float32)
        self.revert_operator = nn.Parameter(torch.from_numpy(revert_opetator))
        self.revert_operator.requires_grad = False
        self.is_visible = is_visible

    def forward(self, Offset, Posmap_kpt):
        offsetmap = Offset * self.offset_scale + self.mean_posmap_tensor
        offsetmap = offsetmap.permute(0, 2, 3, 1)
        kptmap = Posmap_kpt.permute(0, 2, 3, 1)

        offsetmap_np = offsetmap.detach().cpu().numpy()
        for i in range(offsetmap.shape[0]):
            revert_np = np.diagflat([1, -1, 1])
            temp_kptmap = kptmap[i].detach().cpu().numpy()
            offsetmap_np[i] = offsetmap_np[i].dot(revert_np.T)

            srckpt = offsetmap_np[i][uv_kpt[:, 0], uv_kpt[:, 1]]
            dstkpt = temp_kptmap[uv_kpt[:, 0], uv_kpt[:, 1]]
            tform = transform.estimate_transform('similarity', srckpt, dstkpt)
            if not self.is_visible:
                offsetmap_np[i] = offsetmap_np[i].dot(tform.params[0:3, 0:3].T) + tform.params[0:3, 3]
            else:
                R = tform.params[0:3, 0:3].T

                visible_srckpt = []
                visible_dstkpt = []

                yaw_angle = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
                yaw_rate = yaw_angle / np.pi * 0.9
                left = 0 + yaw_rate * 256
                right = 256 + yaw_rate * 256
                if left < 0:
                    left = 0
                if right > 256:
                    right = 256
                for j in range(68):
                    if uv_kpt[j, 1] <= left or uv_kpt[j, 1] >= right:
                        continue
                    else:
                        visible_srckpt.append(srckpt[j])
                        visible_dstkpt.append(dstkpt[j])
                if len(visible_dstkpt) < 10:
                    offsetmap_np[i] = offsetmap_np[i].dot(tform.params[0:3, 0:3].T) + tform.params[0:3, 3]
                else:
                    visible_dstkpt = np.array(visible_dstkpt)
                    visible_srckpt = np.array(visible_srckpt)
                    visible_tform = transform.estimate_transform('similarity', visible_srckpt, visible_dstkpt)
                    offsetmap_np[i] = offsetmap_np[i].dot(visible_tform.params[0:3, 0:3].T) + visible_tform.params[0:3, 3]

        outpos = torch.from_numpy(offsetmap_np).to(self.mean_posmap_tensor.device)
        outpos = outpos.permute(0, 3, 1, 2)
        return outpos


# 根据4组对应点计算旋转矩阵
def vector2Tform_np(p, q):
    A = np.stack([p[0] - p[1], p[0] - p[2], p[0] - p[3]])
    B = np.stack([q[0] - q[1], q[0] - q[2], q[0] - q[3]])
    if np.linalg.det(A) < 1e-4:
        return None, None
    R = np.linalg.inv(A).dot(B)
    T = np.mean(q - p.dot(R), axis=0)
    return R, T


def vector2Tform(p, q):
    A = torch.stack([p[0] - p[1], p[0] - p[2], p[0] - p[3]])
    # if torch.abs(torch.det(A)) < 1e-4:
    #     return None, None
    B = torch.stack([q[0] - q[1], q[0] - q[2], q[0] - q[3]])
    R = torch.inverse(A).mm(B)
    T = torch.mean(q - p.mm(R), dim=0)
    return R, T


class VisibleRebuildModule(nn.Module):
    def __init__(self):
        super(VisibleRebuildModule, self).__init__()
        self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_posmap.transpose((2, 0, 1))))
        self.mean_posmap_tensor.requires_grad = False

        self.S_scale = 1e4
        self.offset_scale = 6
        revert_opetator = np.array([[1., -1., 1.], [1., -1., 1.], [1., -1., 1.]]).astype(np.float32)
        self.revert_operator = nn.Parameter(torch.from_numpy(revert_opetator))
        self.revert_operator.requires_grad = False

    def forward(self, Offset, Posmap_kpt, is_torch=False):
        offsetmap = Offset * self.offset_scale + self.mean_posmap_tensor
        offsetmap = offsetmap.permute(0, 2, 3, 1)
        kptmap = Posmap_kpt.permute(0, 2, 3, 1)
        outpos = torch.zeros((Offset.shape[0], 65536, 3), device=Offset.device)

        kpt_dst = kptmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        kpt_src = offsetmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]

        offsetmap = offsetmap.reshape((Offset.shape[0], 65536, 3))

        if is_torch:
            for i in range(Offset.shape[0]):
                s_coarse = 0
                sr_coarse = torch.zeros((3, 3), device=Offset.device)
                for x in range(0, 64, 4):
                    for y in range(x + 2, x + 3, 4):
                        p = torch.stack([kpt_src[i][x], kpt_src[i][x + 1], kpt_src[i][y], kpt_src[i][y + 1]])
                        q = torch.stack([kpt_dst[i][x], kpt_dst[i][x + 1], kpt_dst[i][y], kpt_dst[i][y + 1]])
                        # temp_R[i][x // 4], temp_T[i][x // 4] = vector2Tform(p, q)
                        r, t = vector2Tform(p, q)
                        if r is None:
                            continue
                        else:
                            sr_coarse = sr_coarse + r
                            s_coarse = s_coarse + 1
                R = sr_coarse / s_coarse

                visibility = torch.ones(68, device=Offset.device)
                yaw_angle = torch.atan2(-R[2, 0], torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
                yaw_rate = yaw_angle / np.pi
                left = 0 + yaw_rate * 256
                right = 256 + yaw_rate * 256
                if left < 0:
                    left = 0
                if right > 256:
                    right = 256
                for j in range(68):
                    if uv_kpt[j, 1] <= left or uv_kpt[j, 1] >= right:
                        visibility[j] = 1e-3
                    else:
                        visibility[j] = 1

                s_precise = 0
                sr_precise = torch.zeros((3, 3), device=Offset.device)
                st_precise = torch.zeros(3, device=Offset.device)
                for x in range(0, 64, 4):
                    for y in range(x + 2, x + 3, 4):
                        p = torch.stack([kpt_src[i][x], kpt_src[i][x + 1], kpt_src[i][y], kpt_src[i][y + 1]])
                        q = torch.stack([kpt_dst[i][x], kpt_dst[i][x + 1], kpt_dst[i][y], kpt_dst[i][y + 1]])
                        r, t = vector2Tform(p, q)
                        if r is None:
                            continue
                        else:
                            weight = visibility[x] * visibility[x + 1] * visibility[y] * visibility[y + 1]
                            s_precise = s_precise + weight
                            sr_precise = sr_precise + weight * r
                            st_precise = st_precise + weight * t

                R_precise = sr_precise / s_precise
                T_precise = st_precise / s_precise
                outpos[i] = offsetmap[i].mm(R_precise) + T_precise

        else:
            for i in range(Offset.shape[0]):
                # s_coarse = 0
                # sr_coarse = np.zeros((3, 3))
                kpt_src_np = kpt_src[i].detach().cpu().numpy()
                kpt_dst_np = kpt_dst[i].detach().cpu().numpy()
                # for x in range(0, 44, 4):
                #     for y in range(x + 10, 63, 1):
                #         p = np.stack([kpt_src_np[x], kpt_src_np[x + 10], kpt_src_np[y], kpt_src_np[y + 5]])
                #         q = np.stack([kpt_dst_np[x], kpt_dst_np[x + 10], kpt_dst_np[y], kpt_dst_np[y + 5]])
                #         # temp_R[i][x // 4], temp_T[i][x // 4] = vector2Tform(p, q)
                #         r, t = vector2Tform_np(p, q)
                #         if r is None:
                #             continue
                #         else:
                #             sr_coarse = sr_coarse + r
                #             s_coarse = s_coarse + 1
                # R = sr_coarse / s_coarse

                visibility = np.ones(68)
                # yaw_angle = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
                # yaw_rate = yaw_angle / np.pi
                # left = 0 + yaw_rate * 256
                # right = 256 + yaw_rate * 256
                # if left < 0:
                #     left = 0
                # if right > 256:
                #     right = 256
                # for j in range(68):
                #     if uv_kpt[j, 1] <= left or uv_kpt[j, 1] >= right:
                #         visibility[j] = 1e-3
                #     else:
                #         visibility[j] = 1

                s_precise = 0
                sr_precise = np.zeros((3, 3))
                st_precise = np.zeros(3)
                for x in range(16, 28, 1):
                    for y in range(48, 68, 1):
                        p = np.stack([kpt_src_np[x], kpt_src_np[x + 20], kpt_src_np[y], kpt_src_np[y // 2 + 5]])
                        q = np.stack([kpt_dst_np[x], kpt_dst_np[x + 20], kpt_dst_np[y], kpt_dst_np[y // 2 + 5]])
                        r, t = vector2Tform_np(p, q)
                        if r is None:
                            continue
                        else:
                            weight = visibility[x] * visibility[x + 20] * visibility[y] * visibility[y // 2 + 5]
                            s_precise = s_precise + weight
                            sr_precise = sr_precise + weight * r
                            st_precise = st_precise + weight * t

                R_precise = torch.from_numpy(sr_precise / s_precise).to(Offset.device).float()
                T_precise = torch.from_numpy(st_precise / s_precise).to(Offset.device).float()
                outpos[i] = offsetmap[i].mm(R_precise) + T_precise

        outpos = outpos.reshape((Offset.shape[0], 256, 256, 3))
        outpos = outpos.permute(0, 3, 1, 2)
        return outpos


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        out = x.view(x.size(0), -1)
        return out


class RTSRegressor(nn.Module):
    def __init__(self, num_cluster=1, filters=512):
        super(RTSRegressor, self).__init__()
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


class QTRegressor(nn.Module):
    def __init__(self, num_cluster=1, filters=512):
        super(QTRegressor, self).__init__()
        self.pipe = nn.Sequential(
            # nn.AvgPool2d(8, stride=1),
            PRNResBlock(in_channels=filters, out_channels=filters * 2, kernel_size=3, stride=2, with_conv_shortcut=True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(2 * filters, 2 * filters, bias=False),
            nn.BatchNorm1d(2 * filters, momentum=0.5),
            nn.ReLU()
            # nn.Linear(2*filters, 2*filters),
            # # nn.BatchNorm1d(2*filters),
            # nn.ReLU()
        )
        self.Q_layer = nn.Sequential(nn.Linear(2 * filters, num_cluster * 4))
        self.T2d_layer = nn.Sequential(nn.Linear(2 * filters, num_cluster * 2))

    def forward(self, x):
        # x_new = x.detach()
        feat = self.pipe(x)
        Q = self.Q_layer(feat)
        T2d = self.T2d_layer(feat)
        return Q, T2d


class AttentionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(AttentionModel, self).__init__()
        self.conv1 = Conv2d_BN_AC(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = Conv2d_BN_AC(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = Conv2d_BN_AC(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = Conv2d_BN_AC(feature_size, feature_size, kernel_size=3, padding=1)

        # self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        # self.act1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(feature_size, 1, kernel_size=3, padding=1, bias=False)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        # x_new = x.detach()
        # out = self.conv1(x)
        # out = self.act1(out)
        # out = self.conv2(out)
        # out = self.act2(out)
        # out = self.conv3(out)
        # out = self.act3(out)
        # out = self.conv4(out)
        # out = self.act4(out)
        # out = self.conv5(out)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out_attention = self.output_act(out)

        # at2 = out_attention.clone()
        # at2[out_attention < 0.2] = 0
        # at2[out_attention > 0.8] = 1
        return out_attention


class MeanQTRegressor(nn.Module):
    def __init__(self, num_cluster=10, filters=512):
        super(MeanQTRegressor, self).__init__()
        self.pipe = nn.Sequential(
            # nn.AvgPool2d(8, stride=1),
            PRNResBlock(in_channels=filters, out_channels=filters * 2, kernel_size=4, stride=2, with_conv_shortcut=True),
            # PRNResBlock(in_channels=filters, out_channels=filters * 2, kernel_size=3, stride=1, with_conv_shortcut=False),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(2 * filters, 2 * filters, bias=False),
            nn.BatchNorm1d(2 * filters, momentum=0.5),
            nn.ReLU()
        )
        self.Q_layer = nn.Linear(2 * filters, num_cluster * 4)
        self.T2d_layer = nn.Linear(2 * filters, num_cluster * 2)
        nn.init.xavier_normal_(self.Q_layer.weight)
        nn.init.xavier_normal_(self.T2d_layer.weight)

    def forward(self, x):
        # x_new = x.detach()
        feat = self.pipe(x)
        Qs = self.Q_layer(feat)
        # Qs = nn.Tanh()(Qs)
        T2ds = self.T2d_layer(feat)
        # T2ds = nn.Tanh()(T2ds)
        return Qs, T2ds
