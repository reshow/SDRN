import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def getAngleList(data_dir):
    angle_list = []
    i = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_name = file.split('.')[0]
            file_type = file.split('.')[1]
            if file_type == 'mat':
                i += 1
                print('\r', root + file_name, i, end='      ')
                m = sio.loadmat(root + '/' + file)
                pose_para = m['Pose_Para'].T.astype(np.float32)
                angles = pose_para[:3, 0]
                angle_list.append(angles)
    return angle_list


def getParamList(data_dir):
    pose_list = []
    shape_list = []
    exp_list = []
    tex_list = []
    color_list = []
    illum_list = []
    i = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_name = file.split('.')[0]
            file_type = file.split('.')[1]
            if file_type == 'mat':
                i += 1
                print('\r', root + file_name, i, end='      ')
                bfm_info = sio.loadmat(root + '/' + file)
                pose_list.append(bfm_info['Pose_Para'].T.astype(np.float32))
                shape_list.append(bfm_info['Shape_Para'].astype(np.float32))
                exp_list.append(bfm_info['Exp_Para'].astype(np.float32))
                tex_list.append(bfm_info['Tex_Para'].astype(np.float32))
                color_list.append(bfm_info['Color_Para'].astype(np.float32))
                illum_list.append(bfm_info['Illum_Para'].astype(np.float32))
    return np.array(pose_list), np.array(shape_list), np.array(exp_list), np.array(tex_list), np.array(color_list), np.array(illum_list)


pl, sl, el, tl, cl, il = getParamList('../data/back/AFLW2000')
# for i in range(7):
#     temp = pl[:, i]
#     max_val = temp.max()
#     min_val = temp.min()
#     step = (max_val - min_val) / 80
#     group = [min_val + step * j for j in range(81)]
#     plt.hist(temp, group, histtype='bar', rwidth=0.8)
#     plt.show()
# for i in range(7):
#     temp = pl[:, i]
#     print(i, temp.min(), temp.max())
#
# pl, sl, el, tl, cl, il = getParamList('../data/back/300W-3D')
# for i in range(7):
#     temp = pl[:, i]
#     print(i, temp.min(), temp.max())

# l1 = getAngleList('../data/back/300W-3D/')
# l2 = getAngleList('../data/back/300W_LP/')
# l3 = getAngleList('../data/back/AFLW2000/')
#
# lnp1 = np.array(l1)
# lnp2 = np.array(l2)
# lnp3 = np.array(l3)
# lnp1 = np.load('l1.npy')
# lnp2 = np.load('l2.npy')
# lnp3 = np.load('l3.npy')
#
# for i in range(2000):
#     for j in range(3):
#         while lnp3[i, j] < -np.pi:
#             lnp3[i, j] += (2 * np.pi)
#         while lnp3[i, j] > np.pi:
#             lnp3[i, j] -= (2 * np.pi)
#
# lx1 = lnp1[:, 0]
# lx2 = lnp2[:, 0]
# lx3 = lnp3[:, 0]
#
# ly1 = lnp1[:, 1]
# ly2 = lnp2[:, 1]
# ly3 = lnp3[:, 1]
#
# lz1 = lnp1[:, 2]
# lz2 = lnp2[:, 2]
# lz3 = lnp3[:, 2]
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
#
# group = [-np.pi + i * np.pi / 40 for i in range(81)]
#
# plt.hist(lx1, group, histtype='bar', rwidth=0.8)
# plt.show()
# plt.hist(lx2, group, histtype='bar', rwidth=0.8)
# plt.show()
# plt.hist(lx3, group, histtype='bar', rwidth=0.8)
# plt.show()
#
# plt.hist(ly1, group, histtype='bar', rwidth=0.8)
# plt.show()
# plt.hist(ly2, group, histtype='bar', rwidth=0.8)
# plt.show()
# plt.hist(ly3, group, histtype='bar', rwidth=0.8)
# plt.show()
#
# plt.hist(lz1, group, histtype='bar', rwidth=0.8)
# plt.show()
# plt.hist(lz2, group, histtype='bar', rwidth=0.8)
# plt.show()
# plt.hist(lz3, group, histtype='bar', rwidth=0.8)
# plt.show()


def crossMat(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def norm(x):
    return x / np.sqrt(x.dot(x))


# 估计出的旋转矩阵只能保证对应向量的旋转是正确的  但沿着旋转后的向量的方向旋转仍然能有无穷多组解
def vector2Rotation(src, dst):
    from math import acos, cos, sin
    p = src / np.sqrt(src.dot(src))
    q = dst / np.sqrt(dst.dot(dst))
    theta = acos(p.dot(q))
    c = p.dot(crossMat(q))
    c = c / np.sqrt(c.dot(c))
    W = crossMat(c)
    I = np.diagflat([1, 1, 1])
    R = I + sin(theta) * W + (1 - cos(theta)) * (W.dot(W))
    s1 = np.sqrt(dst.dot(dst))
    s2 = np.sqrt(src.dot(src))
    R = R * s1 / s2
    return R.T


def planeProjection(u, n):
    n = n / np.sqrt(n.dot(n))
    proj = u - (u.dot(n)) * n
    return proj


# 需要两组向量确定两轮旋转
def doubleVector2Rotation(src, dst, src2, dst2, src3, dst3):
    R1 = vector2Rotation(src, dst)
    wd = src2.dot(R1)

    c = dst / np.sqrt(dst.dot(dst))  # 第二轮旋转的旋转轴为第一次的dst
    wdp = planeProjection(wd, c)
    bn2p = planeProjection(dst2, c)

    R2 = vector2Rotation(wdp, bn2p)

    # c=dst2/np.sqrt(dst.dot(dst2))

    return R1.dot(R2)

#
# class VisibleRebuildModule(nn.Module):
#     def __init__(self):
#         super(VisibleRebuildModule, self).__init__()
#         self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_posmap.transpose((2, 0, 1))))
#         self.mean_posmap_tensor.requires_grad = False
#
#         self.S_scale = 1e4
#         self.offset_scale = 6
#         revert_opetator = np.array([[1., -1., 1.], [1., -1., 1.], [1., -1., 1.]]).astype(np.float32)
#         self.revert_operator = nn.Parameter(torch.from_numpy(revert_opetator))
#         self.revert_operator.requires_grad = False
#
#     def forward(self, Offset, Posmap_kpt, is_torch=False):
#         offsetmap = Offset * self.offset_scale + self.mean_posmap_tensor
#         offsetmap = offsetmap.permute(0, 2, 3, 1)
#         kptmap = Posmap_kpt.permute(0, 2, 3, 1)
#         outpos = torch.zeros((Offset.shape[0], 65536, 3), device=Offset.device)
#
#         kpt_dst = kptmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
#         kpt_src = offsetmap[:, uv_kpt[:, 0], uv_kpt[:, 1]]
#
#         offsetmap = offsetmap.reshape((Offset.shape[0], 65536, 3))
#
#         if is_torch:
#             for i in range(Offset.shape[0]):
#                 s_coarse = 0
#                 sr_coarse = torch.zeros((3, 3), device=Offset.device)
#                 for x in range(0, 64, 4):
#                     for y in range(x + 2, x + 3, 4):
#                         p = torch.stack([kpt_src[i][x], kpt_src[i][x + 1], kpt_src[i][y], kpt_src[i][y + 1]])
#                         q = torch.stack([kpt_dst[i][x], kpt_dst[i][x + 1], kpt_dst[i][y], kpt_dst[i][y + 1]])
#                         # temp_R[i][x // 4], temp_T[i][x // 4] = vector2Tform(p, q)
#                         r, t = vector2Tform(p, q)
#                         if r is None:
#                             continue
#                         else:
#                             sr_coarse = sr_coarse + r
#                             s_coarse = s_coarse + 1
#                 R = sr_coarse / s_coarse
#
#                 visibility = torch.ones(68, device=Offset.device)
#                 yaw_angle = torch.atan2(-R[2, 0], torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
#                 yaw_rate = yaw_angle / np.pi
#                 left = 0 + yaw_rate * 256
#                 right = 256 + yaw_rate * 256
#                 if left < 0:
#                     left = 0
#                 if right > 256:
#                     right = 256
#                 for j in range(68):
#                     if uv_kpt[j, 1] <= left or uv_kpt[j, 1] >= right:
#                         visibility[j] = 1e-3
#                     else:
#                         visibility[j] = 1
#
#                 s_precise = 0
#                 sr_precise = torch.zeros((3, 3), device=Offset.device)
#                 st_precise = torch.zeros(3, device=Offset.device)
#                 for x in range(0, 64, 4):
#                     for y in range(x + 2, x + 3, 4):
#                         p = torch.stack([kpt_src[i][x], kpt_src[i][x + 1], kpt_src[i][y], kpt_src[i][y + 1]])
#                         q = torch.stack([kpt_dst[i][x], kpt_dst[i][x + 1], kpt_dst[i][y], kpt_dst[i][y + 1]])
#                         r, t = vector2Tform(p, q)
#                         if r is None:
#                             continue
#                         else:
#                             weight = visibility[x] * visibility[x + 1] * visibility[y] * visibility[y + 1]
#                             s_precise = s_precise + weight
#                             sr_precise = sr_precise + weight * r
#                             st_precise = st_precise + weight * t
#
#                 R_precise = sr_precise / s_precise
#                 T_precise = st_precise / s_precise
#                 outpos[i] = offsetmap[i].mm(R_precise) + T_precise
#
#         else:
#             for i in range(Offset.shape[0]):
#                 # s_coarse = 0
#                 # sr_coarse = np.zeros((3, 3))
#                 kpt_src_np = kpt_src[i].detach().cpu().numpy()
#                 kpt_dst_np = kpt_dst[i].detach().cpu().numpy()
#                 # for x in range(0, 44, 4):
#                 #     for y in range(x + 10, 63, 1):
#                 #         p = np.stack([kpt_src_np[x], kpt_src_np[x + 10], kpt_src_np[y], kpt_src_np[y + 5]])
#                 #         q = np.stack([kpt_dst_np[x], kpt_dst_np[x + 10], kpt_dst_np[y], kpt_dst_np[y + 5]])
#                 #         # temp_R[i][x // 4], temp_T[i][x // 4] = vector2Tform(p, q)
#                 #         r, t = vector2Tform_np(p, q)
#                 #         if r is None:
#                 #             continue
#                 #         else:
#                 #             sr_coarse = sr_coarse + r
#                 #             s_coarse = s_coarse + 1
#                 # R = sr_coarse / s_coarse
#
#                 visibility = np.ones(68)
#                 # yaw_angle = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
#                 # yaw_rate = yaw_angle / np.pi
#                 # left = 0 + yaw_rate * 256
#                 # right = 256 + yaw_rate * 256
#                 # if left < 0:
#                 #     left = 0
#                 # if right > 256:
#                 #     right = 256
#                 # for j in range(68):
#                 #     if uv_kpt[j, 1] <= left or uv_kpt[j, 1] >= right:
#                 #         visibility[j] = 1e-3
#                 #     else:
#                 #         visibility[j] = 1
#
#                 s_precise = 0
#                 sr_precise = np.zeros((3, 3))
#                 st_precise = np.zeros(3)
#                 for x in range(16, 28, 1):
#                     for y in range(48, 68, 1):
#                         p = np.stack([kpt_src_np[x], kpt_src_np[x + 20], kpt_src_np[y], kpt_src_np[y // 2 + 5]])
#                         q = np.stack([kpt_dst_np[x], kpt_dst_np[x + 20], kpt_dst_np[y], kpt_dst_np[y // 2 + 5]])
#                         r, t = vector2Tform_np(p, q)
#                         if r is None:
#                             continue
#                         else:
#                             weight = visibility[x] * visibility[x + 20] * visibility[y] * visibility[y // 2 + 5]
#                             s_precise = s_precise + weight
#                             sr_precise = sr_precise + weight * r
#                             st_precise = st_precise + weight * t
#
#                 R_precise = torch.from_numpy(sr_precise / s_precise).to(Offset.device).float()
#                 T_precise = torch.from_numpy(st_precise / s_precise).to(Offset.device).float()
#                 outpos[i] = offsetmap[i].mm(R_precise) + T_precise
#
#         outpos = outpos.reshape((Offset.shape[0], 256, 256, 3))
#         outpos = outpos.permute(0, 3, 1, 2)
#         return outpos
