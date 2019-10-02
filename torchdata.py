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
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import augmentation
import matplotlib.pyplot as plt
import visualize

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


def getRotationMatrixFromAxisAngle(x, y, z):
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


def isMatSame(r1, r2, thresh=1e-1):
    diff = np.abs(r1 - r2)
    if (diff < thresh).all():
        return True
    else:
        return False


def estimateRotationAngle(rot_mat):
    sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[0, 1] * rot_mat[0, 1])

    if sy > 1e-6:
        # not singular
        x = atan2(rot_mat[1, 2], rot_mat[2, 2])
        y = atan2(-rot_mat[0, 2], sy)
        z = atan2(rot_mat[0, 1], rot_mat[0, 0])
    else:
        x = atan2(-rot_mat[1, 2], rot_mat[2, 2])
        y = atan2(-rot_mat[0, 2], sy)
        z = 0
    if rot_mat[1, 0] > 1 - 1e-2:
        x = 0
        y = atan2(-rot_mat[0, 2], sy)
        z = atan2(rot_mat[0, 1], rot_mat[0, 0])

    maybe_R = getRotationMatrixFromAxisAngle(x, y, z)
    if isMatSame(rot_mat, maybe_R):
        return x, y, z

    # print(rot_mat)
    return 0, 0, 0


def getMeanPosmap():
    mean_position = bfm.get_mean_shape() * 1e-4
    mean_uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, mean_position, 256, 256, 3)
    np.save('uv-data/mean_uv_posmap.npy', mean_uv_position_map.astype(np.float32))
    return mean_uv_position_map


class ImageData:
    def __init__(self):
        self.cropped_image_path = ''
        self.cropped_posmap_path = ''
        self.init_image_path = ''
        self.init_posmap_path = ''
        self.texture_path = ''
        self.texture_image_path = ''
        self.bbox_info_path = ''
        self.offset_posmap_path = ''

        self.image = None
        self.posmap = None
        self.offset_posmap = None
        self.bbox_info = None
        self.S = None
        self.T = None
        self.R = None

    def readPath(self, image_dir):
        image_name = image_dir.split('/')[-1]
        self.cropped_image_path = image_dir + '/' + image_name + '_cropped.npy'
        self.cropped_posmap_path = image_dir + '/' + image_name + '_cropped_uv_posmap.npy'
        self.init_image_path = image_dir + '/' + image_name + '_init.jpg'
        self.init_posmap_path = image_dir + '/' + image_name + '_uv_posmap.npy'
        # change the format to npy
        self.texture_path = image_dir + '/' + image_name + '_uv_texture_map.npy'
        self.texture_image_path = image_dir + '/' + image_name + '_uv_texture_map.jpg'

        self.bbox_info_path = image_dir + '/' + image_name + '_bbox_info.mat'
        self.offset_posmap_path = image_dir + '/' + image_name + '_offset_posmap.npy'
        # TODO:read bbox and tform

    def readFile(self, mode='posmap'):
        if mode == 'posmap':
            self.image = np.load(self.cropped_image_path).astype(np.uint8)
            self.posmap = np.load(self.cropped_posmap_path).astype(np.float16)
        elif mode == 'offset':
            self.image = np.load(self.cropped_image_path).astype(np.uint8)
            self.posmap = np.load(self.cropped_posmap_path).astype(np.float16)
            self.offset_posmap = np.load(self.offset_posmap_path).astype(np.float16)
            self.bbox_info = sio.loadmat(self.bbox_info_path)
        else:
            pass


def toTensor(image):
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    return image


class DataGenerator(Dataset):
    def __init__(self, all_image_data, mode='posmap', is_aug=False):
        super(DataGenerator, self).__init__()
        self.all_image_data = all_image_data
        self.image_height = 256
        self.image_width = 256
        self.image_channel = 3
        # mode=posmap or offset
        self.mode = mode
        self.is_aug = is_aug

        self.augment = transforms.Compose(
            [
                transforms.ToPILImage(mode='RGB'),
                transforms.RandomOrder(
                    [transforms.RandomGrayscale(p=0.1),
                     transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.25),
                     # transforms.RandomApply([transforms.Lambda(lambda x: augmentation.channelScale(x))], p=0.25),
                     # transforms.RandomApply([transforms.Lambda(lambda x: augmentation.randomErase(x))], p=0.25)
                     ]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        self.toTensor = transforms.ToTensor()
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.no_augment = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        for data in self.all_image_data:
            data.readFile(mode=self.mode)

    def __getitem__(self, index):
        if self.mode == 'posmap':

            image = (self.all_image_data[index].image / 255.0).astype(np.float32)
            pos = self.all_image_data[index].posmap.astype(np.float32)
            if self.is_aug:
                image, pos = augmentation.torchDataAugment(image, pos)
                image = (image * 255.0).astype(np.uint8)
                image = self.augment(image)
                # image=augmentation.prnAugment(image)
                # #image = self.no_augment(image)
            else:
                image = self.no_augment(image)
            pos = pos / 280.
            pos = self.toTensor(pos)
            return image, pos

        elif self.mode == 'offset':

            image = (self.all_image_data[index].image / 255.).astype(np.float32)
            pos = self.all_image_data[index].posmap.astype(np.float32)
            offset = self.all_image_data[index].offset_posmap.astype(np.float32)

            bbox_info = self.all_image_data[index].bbox_info
            trans_mat = bbox_info['TformOffset']

            if self.is_aug:
                if np.random.rand() > 0.75:
                    rot_angle = np.random.randint(-90, 90)
                    rot_angle = rot_angle / 180. * np.pi
                    R_3d, R_3d_inv = augmentation.getRotateMatrix3D(rot_angle, image.shape)
                    trans_mat = R_3d.dot(trans_mat)
                    image, pos = augmentation.rotateData(image, pos, specify_angle=rot_angle)
                image, pos = augmentation.torchDataAugment(image, pos, is_rotate=False)
                image = (image * 255.0).astype(np.uint8)
                image = self.augment(image)
                # 　image = self.no_augment(image)
            else:
                image = self.no_augment(image)

            t0 = trans_mat[0:3, 0]
            S = np.sqrt(np.sum(t0 * t0))
            R = trans_mat[0:3, 0:3]
            R = R.dot(np.diagflat([1 / S, -1 / S, 1 / S]))
            R_flatten = estimateRotationAngle(R)
            R_flatten = np.reshape((np.array(R_flatten)), (3,)) / np.pi
            for i in range(3):
                while R_flatten[i] < -1:
                    R_flatten[i] += 2
                while R_flatten[i] > 1:
                    R_flatten[i] -= 2
            # R_flatten = np.reshape(R, (9,))

            T_flatten = np.reshape(trans_mat[0:3, 3], (3,))
            S = S * 5e2
            T_flatten = T_flatten / 300

            # print(R_flatten)

            # print(R_flatten)
            if S > 1:
                print('too large scale', S)
            if (abs(T_flatten) > 1).any():
                print('too large T', T_flatten)
            if (abs(R_flatten) > 1).any():
                print('too large R', R_flatten)

            R_flatten = torch.from_numpy(R_flatten)
            T_flatten = torch.from_numpy(T_flatten)
            S = torch.tensor(S)

            pos = pos / 280.
            offset = offset / 4.
            pos = self.toTensor(pos)
            offset = self.toTensor(offset)

            return image, pos, offset, R_flatten, T_flatten, S

        else:
            return None

    def __len__(self):
        return len(self.all_image_data)


def getDataLoader(all_image_data, mode='posmap', batch_size=16, is_shuffle=False, is_aug=False):
    dataset = DataGenerator(all_image_data=all_image_data, mode=mode, is_aug=is_aug)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=4, pin_memory=True)
    return train_loader
