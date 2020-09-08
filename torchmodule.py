import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import mean_posmap, uv_kpt
from skimage import io, transform
import visualize

alignment_kpt = np.load('uv-data/alignment_kpt.npy')


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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True), bias=False):
        super(ConvTranspose2d_BN_AC, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, output_padding=stride - 1, bias=bias)

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
        out = out + s
        out = self.BN_AC(out)
        return out


class PRNResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_conv_shortcut=False, activation=nn.ReLU(inplace=True)):
        super(PRNResBlock2, self).__init__()

        if kernel_size % 2 == 1:
            self.pipe = nn.Sequential(
                Conv2d_BN_AC(in_channels=in_channels, out_channels=out_channels // 2, stride=1, kernel_size=1),
                Conv2d_BN_AC(in_channels=out_channels // 2, out_channels=out_channels // 2, stride=stride,
                             kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, stride=1, kernel_size=1, bias=False),

            )
        else:
            self.pipe = nn.Sequential(
                Conv2d_BN_AC(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=1),
                Conv2d_BN_AC(in_channels=out_channels, out_channels=out_channels, stride=stride,
                             kernel_size=kernel_size, padding=kernel_size - 1, padding_mode='circular'),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=1, bias=False)
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
    def __init__(self, is_visible=False):
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
                left = int(0 + yaw_rate * 256)
                right = int(256 + yaw_rate * 256)
                if left < 1:
                    left = 1
                if left > 125:
                    left = 125
                if right > 255:
                    right = 255
                if right < 133:
                    right = 133

                # offsetmap_np[i, :, 0:left, 0] = (offsetmap_np[i, :, 0:left, 0] - offsetmap_np[i, :, 255:255 - left:-1, 0]) / 2
                # offsetmap_np[i, :, 0:left, 1] = (offsetmap_np[i, :, 0:left, 1] + offsetmap_np[i, :, 255:255 - left:-1, 1]) / 2
                # offsetmap_np[i, :, 0:left, 2] = (offsetmap_np[i, :, 0:left, 2] + offsetmap_np[i, :, 255:255 - left:-1, 2]) / 2
                # visualize.show(offsetmap_np[i,:,:,0])

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


# AR+T=B
def kpt2Tform_np(kpt_src, kpt_dst):
    sum_dist1 = np.sum(np.linalg.norm(kpt_src - kpt_src[0], axis=1))
    sum_dist2 = np.sum(np.linalg.norm(kpt_dst - kpt_dst[0], axis=1))
    A = kpt_src * sum_dist2 / sum_dist1
    B = kpt_dst
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.T.dot(BB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # if np.linalg.det(R) < 0:
    #     print('singular R')
    #     Vt[2, :] *= -1
    #     R = Vt.T.dot(U.T)
    t = mu_B - mu_A.dot(R.T)
    return R * sum_dist2 / sum_dist1, t


def kpt2Tform(kpt_src, kpt_dst):
    sum_dist1 = torch.sum(torch.norm(kpt_src - kpt_src[0], dim=1))
    sum_dist2 = torch.sum(torch.norm(kpt_dst - kpt_dst[0], dim=1))
    A = kpt_src * sum_dist2 / sum_dist1
    B = kpt_dst
    mu_A = A.mean(dim=0)
    mu_B = B.mean(dim=0)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.permute(1, 0).mm(BB)
    U, S, V = torch.svd(H)
    R = V.mm(U.permute(1, 0))
    # if torch.det(R) < 0:
    #     print('singular R')
    #     V[1,:] *= -1
    #     R = V.mm(U.permute(1, 0))
    t = torch.mean(B - A.mm(R.permute(1, 0)), dim=0)
    return R * sum_dist2 / sum_dist1, t


def kpt2TformBatch(kpt_src, kpt_dst):
    sum_dist1 = torch.sum(torch.norm(kpt_src - kpt_src[:, 33:34], dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
    sum_dist2 = torch.sum(torch.norm(kpt_dst - kpt_dst[:, 33:34], dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
    A = kpt_src * sum_dist2 / sum_dist1
    B = kpt_dst
    mu_A = A.mean(dim=1, keepdim=True)
    mu_B = B.mean(dim=1, keepdim=True)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.permute(0, 2, 1).matmul(BB)
    U, S, V = torch.svd(H)
    R = V.matmul(U.permute(0, 2, 1))
    t = torch.mean(B - A.matmul(R.permute(0, 2, 1)), dim=1)
    return R * sum_dist2 / sum_dist1, t


def kpt2TformBatchWeighted(kpt_src, kpt_dst, W):
    print(W)
    sum_dist1 = torch.sum(torch.norm(kpt_src - kpt_src[:, 33:34], dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
    sum_dist2 = torch.sum(torch.norm(kpt_dst - kpt_dst[:, 33:34], dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
    A = kpt_src * sum_dist2 / sum_dist1
    B = kpt_dst
    mu_A = A.mean(dim=1, keepdim=True)
    mu_B = B.mean(dim=1, keepdim=True)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.permute(0, 2, 1).matmul(W).matmul(BB)
    U, S, V = torch.svd(H)
    R = V.matmul(U.permute(0, 2, 1))
    t = torch.mean(B - A.matmul(R.permute(0, 2, 1)), dim=1)
    return R * sum_dist2 / sum_dist1, t


def kpt2TformWeighted(kpt_src, kpt_dst, W):
    sum_dist1 = torch.sum(torch.norm(kpt_src - kpt_src[0], dim=1))
    sum_dist2 = torch.sum(torch.norm(kpt_dst - kpt_dst[0], dim=1))
    A = kpt_src * sum_dist2 / sum_dist1
    B = kpt_dst
    mu_A = A.mean(dim=0)
    mu_B = B.mean(dim=0)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.permute(1, 0).mm(W).mm(BB)
    U, S, V = torch.svd(H)
    R = V.mm(U.permute(1, 0))
    # if torch.det(R) < 0:
    #     print('singular R')
    #     V[1,:] *= -1
    #     R = V.mm(U.permute(1, 0))
    t = torch.mean(B - A.mm(R.permute(1, 0)), dim=0)
    return R * sum_dist2 / sum_dist1, t


def kpt2Tform_notorch(kpt_src, kpt_dst):
    kpt_src_np = kpt_src.detach().cpu().numpy()
    kpt_dst_np = kpt_dst.detach().cpu().numpy()
    tform = transform.estimate_transform('similarity', kpt_src_np, kpt_dst_np)
    tform = tform.params
    tform = torch.from_numpy(tform).to(kpt_src.device).float()
    return tform[0:3, 0:3], tform[0:3, 3]


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

    def forward(self, Offset, Posmap_kpt, attention=None, is_kpt=False):
        offsetmap = Offset * self.offset_scale + self.mean_posmap_tensor
        offsetmap = offsetmap.permute(0, 2, 3, 1)
        outpos = torch.zeros((Offset.shape[0], 65536, 3), device=Offset.device)
        # uv_kpt2 = uv_kpt[17:]
        if is_kpt:
            kpt_dst = Posmap_kpt.permute(0, 2, 1)
        else:
            kptmap = Posmap_kpt.permute(0, 2, 3, 1)
            kpt_dst = kptmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        kpt_src = offsetmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        offsetmap = offsetmap.reshape((Offset.shape[0], 65536, 3))

        for i in range(Offset.shape[0]):
            if attention is None:
                R, T = kpt2Tform(kpt_src[i], kpt_dst[i])
            # R, T = kpt2TformWeighted(kpt_src[i], kpt_dst[i])
            else:
                # use attention mask to decide weights
                # center_list = [36, 39, 42, 45, 30, 31, 35, 48, 54, 0, 8, 16]
                # TODO: <0
                attention2 = attention.detach()
                # W = torch.eye(68, device=kpt_src.device)
                W1 = torch.eye(68, device=kpt_src.device)
                W2 = torch.eye(68, device=kpt_src.device)
                shallow_kpt_args = torch.argsort(kpt_dst[:, 2])[34:]
                W2[shallow_kpt_args, shallow_kpt_args] = 20
                # W[center_list, center_list] *= 10
                for j in range(68):
                    t1 = min(max(int(kpt_dst[i, j, 1] / 8 * 280), 0), 31)
                    t2 = min(max(int(kpt_dst[i, j, 0] / 8 * 280), 0), 31)
                    W2[j, j] = W2[j, j] * attention2[i, 0, t1, t2]
                R, T = kpt2TformWeighted(kpt_src[i], kpt_dst[i], W2)

            outpos[i] = offsetmap[i].mm(R.permute(1, 0)) + T

        outpos = outpos.reshape((Offset.shape[0], 256, 256, 3))
        outpos = outpos.permute(0, 3, 1, 2)
        return outpos


class P2RNRebuildModule(nn.Module):
    def __init__(self):
        super(P2RNRebuildModule, self).__init__()
        self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_posmap.transpose((2, 0, 1))))
        self.mean_posmap_tensor.requires_grad = False

        self.S_scale = 1e4
        self.offset_scale = 6
        revert_opetator = np.array([[1., -1., 1.], [1., -1., 1.], [1., -1., 1.]]).astype(np.float32)
        self.revert_operator = nn.Parameter(torch.from_numpy(revert_opetator))
        self.revert_operator.requires_grad = False

    def forward(self, Offset, Posmap_kpt):
        offsetmap = Offset * self.offset_scale + self.mean_posmap_tensor
        offsetmap = offsetmap.permute(0, 2, 3, 1)

        kptmap = Posmap_kpt.permute(0, 2, 3, 1)
        kpt_dst = kptmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        kpt_src = offsetmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]

        # kpt_dst = kptmap[:, alignment_kpt[:, 0], alignment_kpt[:, 1]]
        # kpt_src = offsetmap[:, alignment_kpt[:, 0], alignment_kpt[:, 1]]
        # Weight = vis_map[:, alignment_kpt[:, 0], alignment_kpt[:, 1]]
        offsetmap = offsetmap.reshape((Offset.shape[0], 65536, 3))
        R, T = kpt2TformBatch(kpt_src, kpt_dst)
        outpos = offsetmap.matmul(R.permute(0, 2, 1)) + T.unsqueeze(1)
        outpos = outpos.reshape((Offset.shape[0], 256, 256, 3))
        outpos = outpos.permute(0, 3, 1, 2)
        return outpos


def calculateVisibility(posmap):
    B, C, W, H = posmap.shape
    down_shift = torch.zeros((B, C, W, H), device=posmap.device)
    right_shift = torch.zeros((B, C, W, H), device=posmap.device)
    down_shift[:, :, :, 1:] = posmap[:, :, :, :-1]
    right_shift[:, :, 1:, :] = posmap[:, :, :-1, :]
    ab = posmap - right_shift
    bc = down_shift - right_shift
    z = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    z[z > 0] = 1
    z[z < 0] = 0.1
    return z


class P2RNVisibilityRebuildModule(nn.Module):
    def __init__(self):
        super(P2RNVisibilityRebuildModule, self).__init__()
        self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_posmap.transpose((2, 0, 1))))
        self.mean_posmap_tensor.requires_grad = False

        self.S_scale = 1e4
        self.offset_scale = 6
        revert_opetator = np.array([[1., -1., 1.], [1., -1., 1.], [1., -1., 1.]]).astype(np.float32)
        self.revert_operator = nn.Parameter(torch.from_numpy(revert_opetator))
        self.revert_operator.requires_grad = False

    def forward(self, Offset, Posmap_kpt, attention=None):
        B, C, W, H = Posmap_kpt.shape
        offsetmap = Offset * self.offset_scale + self.mean_posmap_tensor
        offsetmap = offsetmap.permute(0, 2, 3, 1)

        vis_map = calculateVisibility(Posmap_kpt)

        kptmap = Posmap_kpt.permute(0, 2, 3, 1)
        kpt_dst = kptmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        kpt_src = offsetmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        Weight = vis_map[:, uv_kpt[:, 0], uv_kpt[:, 1]]

        if attention is not None:
            kpt_ind = (kpt_dst[:, :, :2] * 280).long()
            Weight_at = torch.stack([attention[i, 0, kpt_ind[i, :, 1] // 8, kpt_ind[i, :, 0] // 8] for i in range(B)])
            Weight = Weight * Weight_at

        # print(Weight)

        offsetmap = offsetmap.reshape((B, W * H, C))
        R, T = kpt2TformBatchWeighted(kpt_src, kpt_dst, torch.diag_embed(Weight))
        outpos = offsetmap.matmul(R.permute(0, 2, 1)) + T.unsqueeze(1)
        outpos = outpos.reshape((B, W, H, C))
        outpos = outpos.permute(0, 3, 1, 2)
        return outpos


class VisibleRebuildModuleNoOffset(nn.Module):
    def __init__(self):
        super(VisibleRebuildModuleNoOffset, self).__init__()
        revert_opetator = np.array([[1., -1., 1.], [1., -1., 1.], [1., -1., 1.]]).astype(np.float32)
        self.revert_operator = nn.Parameter(torch.from_numpy(revert_opetator))
        self.revert_operator.requires_grad = False

    def forward(self, Offset, Posmap_kpt, is_torch=False):
        offsetmap = Offset
        offsetmap = offsetmap.permute(0, 2, 3, 1)
        kptmap = Posmap_kpt.permute(0, 2, 3, 1)
        outpos = torch.zeros((Offset.shape[0], 65536, 3), device=Offset.device)
        kpt_dst = kptmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        kpt_src = offsetmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
        offsetmap = offsetmap.reshape((Offset.shape[0], 65536, 3))

        for i in range(Offset.shape[0]):
            R, T = kpt2Tform(kpt_src[i], kpt_dst[i])
            outpos[i] = offsetmap[i].mm(R.permute(1, 0)) + T

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
        # self.R_layer = nn.Sequential(nn.Linear(filters, num_cluster * 3), nn.Sigmoid())
        self.R_layer = nn.Sequential(nn.Linear(filters, filters // 2), nn.Linear(filters // 2, num_cluster * 3))
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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv4x4(in_planes, out_planes, stride=1, padding=3, dilation=1, padding_mode='circular'):
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=padding, bias=False, dilation=dilation, padding_mode=padding_mode)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_conv_shortcut=False, dilation=1):
        super(ResBlock, self).__init__()

        norm_layer = nn.BatchNorm2d
        expansion = 2

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, out_channels // expansion)
        self.bn1 = norm_layer(out_channels // expansion)
        self.conv2 = conv3x3(out_channels // expansion, out_channels // expansion, stride=stride, dilation=dilation)
        self.bn2 = norm_layer(out_channels // expansion)
        self.conv3 = conv1x1(out_channels // expansion, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if with_conv_shortcut:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        return out


class ResBlock4(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_conv_shortcut=False, dilation=1):
        super(ResBlock4, self).__init__()

        norm_layer = nn.BatchNorm2d
        expansion = 2

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, out_channels // expansion)
        self.bn1 = norm_layer(out_channels // expansion, momentum=0.5)
        self.conv2 = conv4x4(out_channels // expansion, out_channels // expansion, stride=stride)
        self.bn2 = norm_layer(out_channels // expansion, momentum=0.5)
        self.conv3 = conv1x1(out_channels // expansion, out_channels)
        self.bn3 = norm_layer(out_channels, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        if with_conv_shortcut:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels, momentum=0.5),
            )
        else:
            self.shortcut = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        return out
