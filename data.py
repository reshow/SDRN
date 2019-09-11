import os
import sys
import numpy as np
import scipy.io as sio
from skimage import io, transform
import time
import math
import skimage
import faceutil
from faceutil import mesh
from faceutil.morphable_model import MorphabelModel
from matlabutil import NormDirection
import multiprocessing
import matplotlib.pyplot as plt
import argparse
import ast
import copy
from PIL import ImageEnhance, ImageOps, ImageFile, Image
import augmentation
import random

#  global data
bfm = MorphabelModel('data/Out/BFM.mat')
default_init_image_shape = np.array([450, 450, 3])
default_cropped_image_shape = np.array([256, 256, 3])
default_uvmap_shape = np.array([256, 256, 3])
face_mask_np = io.imread('uv-data/uv_face_mask.png') / 255.
face_mask_mean_fix_rate = (256 * 256) / np.sum(face_mask_np)


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
                 'colors': tex_color, 'landmarks': bfm_info['pt3d_68'].T}
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


class ImageData:
    def __init__(self):
        self.cropped_image_path = ''
        self.cropped_posmap_path = ''
        self.init_image_path = ''
        self.init_posmap_path = ''
        self.texture_path = ''
        self.texture_image_path = ''
        self.bbox_info_path = ''

    def readPath(self, image_dir):
        image_name = image_dir.split('/')[-1]
        self.cropped_image_path = image_dir + '/' + image_name + '_cropped.jpg'
        self.cropped_posmap_path = image_dir + '/' + image_name + '_cropped_uv_posmap.npy'
        self.init_image_path = image_dir + '/' + image_name + '_init.jpg'
        self.init_posmap_path = image_dir + '/' + image_name + '_uv_posmap.npy'
        # change the format to npy
        self.texture_path = image_dir + '/' + image_name + '_uv_texture_map.npy'
        self.texture_image_path = image_dir + '/' + image_name + '_uv_texture_map.jpg'

        self.bbox_info_path = image_dir + '/' + image_name + '_bbox_info.mat'
        # TODO:read bbox and tform


class FitGenerator:
    def __init__(self, all_image_data):
        self.all_image_data = all_image_data
        self.next_index = 0
        self.image_height = 256
        self.image_width = 256
        self.image_channel = 3

    def gen(self, batch_size=64, gen_mode='random'):
        """

        :param batch_size:
        :param gen_mode: random or order
        :return:
        """
        while True:
            x = []
            y = []
            if gen_mode == 'random':
                batch_num = batch_size
                indexs = np.random.randint(len(self.all_image_data), size=batch_size)
            elif gen_mode == 'order':
                if self.next_index == 0:
                    # print('random shuffle')
                    random.shuffle(self.all_image_data)
                if batch_size > len(self.all_image_data):
                    batch_size = len(self.all_image_data)
                batch_num = batch_size
                if self.next_index + batch_size >= len(self.all_image_data):
                    batch_num = len(self.all_image_data) - self.next_index
                indexs = np.array(range(self.next_index, self.next_index + batch_num))
                # print(self.next_index,self.next_index+batch_num)
                self.next_index = (self.next_index + batch_num) % len(self.all_image_data)
            else:
                indexs = None
                batch_num = 0
                print('unknown generate mode')
            for i in range(batch_num):
                image_path = self.all_image_data[indexs[i]].cropped_image_path
                image = io.imread(image_path) / 255.
                image = transform.resize(image, (self.image_height, self.image_width, self.image_channel))
                pos_path = self.all_image_data[indexs[i]].cropped_posmap_path
                pos = np.load(pos_path)
                pos = pos / 256.
                x.append(image)
                y.append(pos)
            x = np.array(x)
            y = np.array(y)
            yield x, y
    def get(self, batch_size=64, gen_mode='random'):
        """

        :param batch_size:
        :param gen_mode: random or order
        :return:
        """

        x = []
        y = []
        if gen_mode == 'random':
            batch_num = batch_size
            indexs = np.random.randint(len(self.all_image_data), size=batch_size)
        elif gen_mode == 'order':
            if self.next_index == 0:
                # print('random shuffle')
                random.shuffle(self.all_image_data)
            if batch_size > len(self.all_image_data):
                batch_size = len(self.all_image_data)
            batch_num = batch_size
            if self.next_index + batch_size >= len(self.all_image_data):
                batch_num = len(self.all_image_data) - self.next_index
            indexs = np.array(range(self.next_index, self.next_index + batch_num))
            # print(self.next_index,self.next_index+batch_num)
            self.next_index = (self.next_index + batch_num) % len(self.all_image_data)
        else:
            indexs = None
            batch_num = 0
            print('unknown generate mode')
        for i in range(batch_num):
            image_path = self.all_image_data[indexs[i]].cropped_image_path
            image = io.imread(image_path) / 255.
            image = transform.resize(image, (self.image_height, self.image_width, self.image_channel))
            pos_path = self.all_image_data[indexs[i]].cropped_posmap_path
            pos = np.load(pos_path)
            pos = pos / 256.
            x.append(image)
            y.append(pos)
        x = np.array(x)
        y = np.array(y)
        return x,y
