import os
import sys
import numpy as np
import scipy.io as sio
from skimage import io
import time
import math
import skimage
import faceutil
from faceutil import mesh
from faceutil.morphable_model import MorphabelModel
from matlabutil import NormDirection
from math import sin, cos, asin, acos, atan, atan2
from PIL import Image
import matplotlib.pyplot as plt

#  global data
bfm = MorphabelModel('data/Out/BFM.mat')
default_init_image_shape = np.array([450, 450, 3])
default_cropped_image_shape = np.array([256, 256, 3])
default_uvmap_shape = np.array([256, 256, 3])
face_mask_np = io.imread('uv-data/uv_face_mask.png') / 255.
face_mask_mean_fix_rate = (256 * 256) / np.sum(face_mask_np)
mean_posmap = np.load('uv-data/mean_uv_posmap.npy')


def process_uv(uv_coordinates):
    [uv_h, uv_w, uv_c] = default_uvmap_shape
    uv_coordinates[:, 0] = uv_coordinates[:, 0] * (uv_w - 1)
    uv_coordinates[:, 1] = uv_coordinates[:, 1] * (uv_h - 1)
    uv_coordinates[:, 1] = uv_h - uv_coordinates[:, 1] - 1
    uv_coordinates = np.hstack((uv_coordinates, np.zeros((uv_coordinates.shape[0], 1))))  # add z
    return uv_coordinates


def readUVKpt(uv_kpt_path):
    file = open(uv_kpt_path, 'r', encoding='utf-8')
    lines = file.readlines()
    # txt is inversed
    x_line = lines[1]
    y_line = lines[0]
    uv_kpt = np.zeros((68, 2)).astype(int)
    x_tokens = x_line.strip().split(' ')
    y_tokens = y_line.strip().split(' ')
    for i in range(68):
        uv_kpt[i][0] = int(float(x_tokens[i]))
        uv_kpt[i][1] = int(float(y_tokens[i]))
    return uv_kpt


#  global data
uv_coords = faceutil.morphable_model.load.load_uv_coords('data/Out/BFM_UV.mat')
uv_coords = process_uv(uv_coords)
uv_kpt = readUVKpt('uv-data/uv_kpt_ind.txt')
uvmap_place_holder = np.ones((256, 256, 1))


def getLandmark(ipt):
    # from uv map
    kpt = ipt[uv_kpt[:, 0], uv_kpt[:, 1]]
    return kpt


def bfm2Mesh(bfm_info, image_shape=default_init_image_shape):
    [image_h, image_w, channel] = image_shape
    pose_para = bfm_info['Pose_Para'].T.astype(np.float32)
    shape_para = bfm_info['Shape_Para'].astype(np.float32)
    exp_para = bfm_info['Exp_Para'].astype(np.float32)
    tex_para = bfm_info['Tex_Para'].astype(np.float32)
    color_Para = bfm_info['Color_Para'].astype(np.float32)
    illum_Para = bfm_info['Illum_Para'].astype(np.float32)

    # 2. generate mesh_numpy
    # shape & exp param
    vertices = bfm.generate_vertices(shape_para, exp_para)
    # texture param
    tex = bfm.generate_colors(tex_para)
    norm = NormDirection(vertices, bfm.model['tri'])

    # color param
    [Gain_r, Gain_g, Gain_b, Offset_r, Offset_g, Offset_b, c] = color_Para[0]
    M = np.array([[0.3, 0.59, 0.11], [0.3, 0.59, 0.11], [0.3, 0.59, .11]])

    g = np.diag([Gain_r, Gain_g, Gain_b])
    o = [Offset_r, Offset_g, Offset_b]
    o = np.tile(o, (vertices.shape[0], 1))

    # illum param
    [Amb_r, Amb_g, Amb_b, Dir_r, Dir_g, Dir_b, thetal, phil, ks, v] = illum_Para[0]
    Amb = np.diag([Amb_r, Amb_g, Amb_b])
    Dir = np.diag([Dir_r, Dir_g, Dir_b])
    l = np.array([math.cos(thetal) * math.sin(phil), math.sin(thetal), math.cos(thetal) * math.cos(phil)]).T
    h = l + np.array([0, 0, 1]).T
    h = h / math.sqrt(h.T.dot(h))

    # final color
    n_l = l.T.dot(norm.T)
    n_h = h.T.dot(norm.T)
    n_l = np.array([max(x, 0) for x in n_l])
    n_h = np.array([max(x, 0) for x in n_h])
    n_l = np.tile(n_l, (3, 1))
    n_h = np.tile(n_h, (3, 1))
    L = Amb.dot(tex.T) + Dir.dot(n_l * tex.T) + (ks * Dir).dot((n_h ** v))
    CT = g.dot(c * np.eye(3) + (1 - c) * M)
    tex_color = CT.dot(L) + o.T
    tex_color = np.minimum(np.maximum(tex_color, 0), 1).T

    # transform mesh_numpy
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]

    # 3ddfa-R: radian || normal transform - R:degree
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    # should not -1
    image_vertices[:, 1] = image_h - image_vertices[:, 1]
    mesh_info = {'vertices': image_vertices, 'triangles': bfm.full_triangles,
                 'full_triangles': bfm.full_triangles,
                 'colors': tex_color}
    # 'landmarks': bfm_info['pt3d_68'].T
    return mesh_info


def UVmap2Mesh(uv_position_map, uv_texture_map=None, only_foreface=True):
    """
    if no texture map is provided, translate the position map to a point cloud
    :param uv_position_map:
    :param uv_texture_map:
    :param only_foreface:
    :return:
    """
    [uv_h, uv_w, uv_c] = default_uvmap_shape
    vertices = []
    colors = []
    triangles = []
    if uv_texture_map is not None:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(uv_texture_map[i][j])
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    pd = (i + 1) * uv_h + j + 1
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                        triangles.append([pa, pc, pb])
                        triangles.append([pa, pb, pd])
                        triangles.append([pa, pd, pb])
                else:
                    if face_mask_np[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(uv_texture_map[i][j])
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        pd = (i + 1) * uv_h + j + 1
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if not face_mask_np[i, j + 1] == 0:
                                if not face_mask_np[i - 1, j] == 0:
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])
                                if not face_mask_np[i + 1, j + 1] == 0:
                                    triangles.append([pa, pb, pd])
                                    triangles.append([pa, pd, pb])
    else:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(np.array([128, 0, 128]))
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                else:
                    if face_mask_np[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(np.array([128, 0, 128]))
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if not face_mask_np[i, j + 1] == 0:
                                if not face_mask_np[i - 1, j] == 0:
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])

    vertices = np.array(vertices)
    colors = np.array(colors)
    triangles = np.array(triangles)
    # verify_face = mesh.render.render_colors(verify_vertices, verify_triangles, verify_colors, height, width,
    #                                         channel)
    mesh_info = {'vertices': vertices, 'triangles': triangles,
                 'full_triangles': triangles,
                 'colors': colors}
    return mesh_info


def mesh2UVmap(mesh_data):
    [uv_h, uv_w, uv_c] = default_uvmap_shape
    vertices = mesh_data['vertices']
    colors = mesh_data['colors']
    triangles = mesh_data['full_triangles']
    # colors = colors / np.max(colors)
    # model_image = mesh.render.render_colors(vertices, bfm.triangles, colors, image_h, image_w) # only for show

    uv_texture_map = mesh.render.render_colors(uv_coords, triangles, colors, uv_h, uv_w, uv_c)
    position = vertices.copy()
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
    uv_position_map = mesh.render.render_colors(uv_coords, triangles, position, uv_h, uv_w, uv_c)
    return uv_position_map, uv_texture_map


def renderMesh(mesh_info, image_shape=None):
    if image_shape is None:
        image_height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
        image_width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    else:
        [image_height, image_width, image_channel] = image_shape
    mesh_image = mesh.render.render_colors(mesh_info['vertices'],
                                           mesh_info['triangles'],
                                           mesh_info['colors'], image_height, image_width)
    mesh_image = np.clip(mesh_image, 0., 1.)
    return mesh_image


def getTransformMatrix(s, angles, t, height):
    """

    :param s: scale
    :param angles: [3] rad
    :param t: [3]
    :return: 4x4 transmatrix
    """
    x, y, z = angles[0], angles[1], angles[2]

    Rx = np.array([[1, 0, 0],
                   [0, cos(x), sin(x)],
                   [0, -sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, -sin(y)],
                   [0, 1, 0],
                   [sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), sin(z), 0],
                   [-sin(z), cos(z), 0],
                   [0, 0, 1]])
    # rotate
    R = Rx.dot(Ry).dot(Rz)
    R = R.astype(np.float32)
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[3, 3] = 1.
    # scale
    S = np.diagflat([s, s, s, 1.])
    T = S.dot(T)
    # offset move
    M = np.diagflat([1., 1., 1., 1.])
    M[0:3, 3] = t.astype(np.float32)
    T = M.dot(T)
    # revert height
    # x[:,1]=height-x[:,1]
    H = np.diagflat([1., 1., 1., 1.])
    H[1, 1] = -1.0
    H[1, 3] = height
    T = H.dot(T)
    return T.astype(np.float32)


def isMatSame(r1, r2, thresh=1e-1):
    diff = np.abs(r1 - r2)
    if (diff < thresh).all():
        return True
    else:
        return False


def angle2Matrix(x, y, z):
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), sin(x)],
                   [0, -sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, -sin(y)],
                   [0, 1, 0],
                   [sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), sin(z), 0],
                   [-sin(z), cos(z), 0],
                   [0, 0, 1]])
    # rotate
    R = Rx.dot(Ry).dot(Rz)
    R = R.astype(np.float32)
    # print(R)
    return R


def angle2Quaternion(y, p, r):
    # yaw pitch roll = x y z
    Z = sin(r / 2) * cos(p / 2) * cos(y / 2) - cos(r / 2) * sin(p / 2) * sin(y / 2)
    Y = cos(r / 2) * sin(p / 2) * cos(y / 2) + sin(r / 2) * cos(p / 2) * sin(y / 2)
    X = cos(r / 2) * cos(p / 2) * sin(y / 2) - sin(r / 2) * sin(p / 2) * cos(y / 2)
    W = cos(r / 2) * cos(p / 2) * cos(y / 2) + sin(r / 2) * sin(p / 2) * sin(y / 2)

    norm_factor = np.sqrt(W * W + X * X + Y * Y + Z * Z)
    W, X, Y, Z = (W / norm_factor, X / norm_factor, Y / norm_factor, Z / norm_factor)
    return W, X, Y, Z


def matrix2Quaternion(R, is_normalize=True):
    if 1 + R[0, 0] + R[1, 1] + R[2, 2] < 0:
        return 1, 0, 0, 0
    q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    if q0 < 1e-8:
        q0 = 1e-8
    q1 = (R[1, 2] - R[2, 1]) / (4 * q0)
    q2 = (R[2, 0] - R[0, 2]) / (4 * q0)
    q3 = (R[0, 1] - R[1, 0]) / (4 * q0)
    if is_normalize:
        norm_factor = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0, q1, q2, q3 = (q0 / norm_factor, q1 / norm_factor, q2 / norm_factor, q3 / norm_factor)
    return q0, q1, q2, q3


def matrix2Angle(R):
    cosy = np.sqrt(R[0, 0] * R[0, 0] + R[0, 1] * R[0, 1])

    if cosy > 1e-6 or cosy < -1e-6:
        # not singular
        x = atan2(R[1, 2], R[2, 2])
        y = atan2(-R[0, 2], cosy)
        z = atan2(R[0, 1], R[0, 0])
    else:
        if R[0, 2] > 0.9:
            x = atan2(-R[1, 2], -R[2, 2])
            y = -np.pi / 2
        else:
            x = atan2(R[1, 2], R[2, 2])
            y = np.pi / 2
        z = 0  # can be any angle

    # maybe_R = getRotationMatrixFromAxisAngle(x, y, z)
    # if isMatSame(rot_mat, maybe_R):
    return x, y, z


def quaternion2Matrix(Q, is_normalize=False):
    q0, q1, q2, q3 = Q
    if is_normalize:
        norm_factor = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0, q1, q2, q3 = (q0 / norm_factor, q1 / norm_factor, q2 / norm_factor, q3 / norm_factor)
    R = [[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
         [2 * (q1 * q2 - q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q0 * q1 + q2 * q3)],
         [2 * (q0 * q2 + q1 * q3), 2 * (q2 * q3 - q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]]
    return np.array(R)


def getMeanPosmap():
    mean_position = bfm.get_mean_shape() * 1e-4
    mean_uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, mean_position, 256, 256, 3)
    np.save('uv-data/mean_uv_posmap.npy', mean_uv_position_map.astype(np.float32))
    return mean_uv_position_map


def getColors(image, posmap):
    [h, w, _] = image.shape
    [uv_h, uv_w, uv_c] = posmap.shape
    # tex = np.zeros((uv_h, uv_w, uv_c))
    around_posmap = np.around(posmap).clip(0, h - 1).astype(np.int)

    tex = image[around_posmap[:, :, 1], around_posmap[:, :, 0], :]
    return tex


def getNewKptMap():
    triangles = bfm.full_triangles
    weight_factor_list = []
    for i in range(68):
        colors = np.zeros((53215, 1))
        colors[bfm.kpt_ind[i], 0] = 100
        uv_texture_map = mesh.render.render_colors(uv_coords, triangles, colors, 256, 256, 1).squeeze()
        temp_weight_factor = []
        for x in range(256):
            for y in range(256):
                if uv_texture_map[x, y] > 0:
                    temp_weight_factor.append([x, y, uv_texture_map[x, y]])
        if len(temp_weight_factor) == 0:
            temp_weight_factor.append([uv_kpt[i, 0], uv_kpt[i, 1], 1])
        temp_weight_factor = np.array(temp_weight_factor)
        total_weight = np.sum(temp_weight_factor[:, 2])
        temp_weight_factor[:, 2] = temp_weight_factor[:, 2] / total_weight
        weight_factor_list.append(temp_weight_factor)

    kpt_mask = np.zeros((256, 256))
    for temp_weight_factor in weight_factor_list:
        for factor in temp_weight_factor:
            kpt_mask[int(factor[0]), int(factor[1])] = factor[2]
    return weight_factor_list, kpt_mask


fl, fm = getNewKptMap()


def getWeightedKpt(pos):
    kpt = []
    for i in range(len(fl)):
        p = np.zeros(3)
        for j in range(len(fl[i])):
            p += pos[int(fl[i][j][0]), int(fl[i][j][1])] * fl[i][j][2]
        kpt.append(p)
    kpt=np.asarray(kpt)
    return kpt
