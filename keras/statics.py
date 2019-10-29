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
for i in range(7):
    temp = pl[:, i]
    print(i, temp.min(), temp.max())

pl, sl, el, tl, cl, il = getParamList('../data/back/300W-3D')
for i in range(7):
    temp = pl[:, i]
    print(i, temp.min(), temp.max())

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