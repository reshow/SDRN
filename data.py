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
from datautil import NormDirection
import multiprocessing
import matplotlib.pyplot as plt
import argparse
import ast
import copy
from PIL import ImageEnhance, ImageOps, ImageFile, Image


class ImageData:
    def __init__(self):
        self.x_path = ''
        self.y_path = ''
        self.init_x_path = ''
        self.init_y_path = ''
        self.texture_path = ''
        self.bbox_info_path = ''
        # 2D bbox
        self.bbox_size = -1
        self.init_bbox_size = -1
        self.tform = None
        #

    def readPath(self, image_dir):
        image_name = image_dir.split('/')[-1]
        self.x_path = image_dir + '/' + image_name + '_cropped.jpg'
        self.y_path = image_dir + '/' + image_name + '_cropped_uv_posmap.npy'
        self.init_x_path = image_dir + '/' + image_name + '_init.jpg'
        self.init_y_path = image_dir + '/' + image_name + '_uv_posmap.npy'
        # change the format to npy
        self.texture_path = image_dir + '/' + image_name + '_uv_texture_map.npy'
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
                if batch_size > len(self.all_image_data):
                    batch_size = len(self.all_image_data)
                batch_num = batch_size
                if self.next_index + batch_size >= len(self.all_image_data):
                    batch_num = len(self.all_image_data) - self.next_index
                indexs = np.array(range(self.next_index, self.next_index + batch_num))
                self.next_index = (self.next_index + batch_num) % len(self.all_image_data)
            else:
                indexs = None
                batch_num = 0
                print('known generate mode')
            for i in range(batch_num):
                image_path = self.all_image_data[indexs[i]].x_path
                image = io.imread(image_path) / 255.
                image = transform.resize(image, (self.image_height, self.image_width, self.image_channel))
                pos_path = self.all_image_data[indexs[i]].y_path
                pos = np.load(pos_path)
                if np.random.rand() > 0.3:
                    image, pos = DataAugmente.rotateData(image, pos)
                    image = DataAugmente.randomColor(image)
                    image = DataAugmente.gaussNoise(image)
                pos = pos / 256.
                x.append(image)
                y.append(pos)
            x = np.array(x)
            y = np.array(y)
            yield x, y


class DataProcessor:
    @staticmethod
    def process_uv(uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
        return uv_coords

    def __init__(self, bfm_path='data/Out/BFM.mat', bfmuv_path='data/Out/BFM_UV.mat',
                 is_full_image=False, is_visualize=True,
                 bbox_extend_rate=1.5, marg_rate=0.1, foreface_path='uv-data/uv_face_mask.png'):
        self.bfm = MorphabelModel(bfm_path)
        print('bfm model loaded')
        self.uv_h = self.uv_w = 256
        self.uv_coords = faceutil.morphable_model.load.load_uv_coords(bfmuv_path)
        self.uv_coords = self.process_uv(self.uv_coords, self.uv_h, self.uv_w)

        self.image_file_name = ''
        self.image_name = ''
        self.image_path = ''
        self.image_dir = ''
        self.output_dir = ''
        self.write_dir = ''
        self.init_image = None
        self.image_shape = None
        self.bfm_info = None
        self.uv_position_map = None
        self.uv_texture_map = None
        self.mesh_info = None

        self.is_full_image = is_full_image
        self.is_visualize = is_visualize,
        self.bbox_extend_rate = bbox_extend_rate
        self.marg_rate = marg_rate

        face_mask = io.imread(foreface_path) / 255.
        # face_mask = np.reshape(face_mask, (256, 256, 1))
        # face_mask = np.concatenate([face_mask] * 3, 2)
        [image_h, image_w] = face_mask.shape
        self.face_mask_mean_fix_rate = (image_h * image_w) / np.sum(face_mask)
        self.face_mask_np = face_mask

    def bfm2Mesh(self, bfm_info=None):
        # check
        if bfm_info is None:
            assert (self.bfm_info is not None)
            bfm_info = self.bfm_info
        assert (self.image_shape is not None)

        [image_h, image_w, channel] = self.image_shape
        pose_para = bfm_info['Pose_Para'].T.astype(np.float32)
        shape_para = bfm_info['Shape_Para'].astype(np.float32)
        exp_para = bfm_info['Exp_Para'].astype(np.float32)
        tex_para = bfm_info['Tex_Para'].astype(np.float32)
        color_Para = bfm_info['Color_Para'].astype(np.float32)
        illum_Para = bfm_info['Illum_Para'].astype(np.float32)

        # 2. generate mesh_numpy
        # shape & exp param
        vertices = self.bfm.generate_vertices(shape_para, exp_para)
        # texture param
        tex = self.bfm.generate_colors(tex_para)
        norm = NormDirection(vertices, self.bfm.model['tri'])

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
        transformed_vertices = self.bfm.transform_3ddfa(vertices, s, angles, t)
        projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
        image_vertices = projected_vertices.copy()
        # should not -1
        image_vertices[:, 1] = image_h - image_vertices[:, 1]
        mesh_data = {'vertices': image_vertices, 'triangles': self.bfm.full_triangles,
                     'full_triangles': self.bfm.full_triangles,
                     'colors': tex_color, 'landmarks': bfm_info['pt3d_68'].T}
        self.mesh_info = mesh_data
        return self.mesh_info

    def UVmap2Mesh(self, uv_position_map=None, uv_texture_map=None, only_foreface=True):
        if uv_position_map is None:
            assert (self.uv_position_map is not None)
            uv_position_map = self.uv_position_map
        if uv_texture_map is None:
            assert (self.uv_texture_map is not None)
            uv_texture_map = self.uv_texture_map
        verify_vertices = []
        verify_colors = []
        verify_triangles = []
        for i in range(self.uv_h):
            for j in range(self.uv_w):
                if not only_foreface:
                    verify_vertices.append(uv_position_map[i][j])
                    verify_colors.append(uv_texture_map[i][j])
                    pa = i * self.uv_h + j
                    pb = i * self.uv_h + j + 1
                    pc = (i - 1) * self.uv_h + j
                    pd = (i + 1) * self.uv_h + j + 1
                    if (i > 0) & (i < self.uv_h - 1) & (j < self.uv_w - 1):
                        verify_triangles.append([pa, pb, pc])
                        verify_triangles.append([pa, pc, pb])
                        verify_triangles.append([pa, pb, pd])
                        verify_triangles.append([pa, pd, pb])
                else:
                    if self.face_mask_np[i, j] == 0:
                        verify_vertices.append(np.array([0, 0, 0]))
                        verify_colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        verify_vertices.append(uv_position_map[i][j])
                        verify_colors.append(uv_texture_map[i][j])
                        pa = i * self.uv_h + j
                        pb = i * self.uv_h + j + 1
                        pc = (i - 1) * self.uv_h + j
                        pd = (i + 1) * self.uv_h + j + 1
                        if (i > 0) & (i < self.uv_h - 1) & (j < self.uv_w - 1):
                            if not self.face_mask_np[i, j + 1] == 0:
                                if not self.face_mask_np[i - 1, j] == 0:
                                    verify_triangles.append([pa, pb, pc])
                                    verify_triangles.append([pa, pc, pb])
                                if not self.face_mask_np[i + 1, j + 1] == 0:
                                    verify_triangles.append([pa, pb, pd])
                                    verify_triangles.append([pa, pd, pb])

        verify_vertices = np.array(verify_vertices)
        verify_colors = np.array(verify_colors)
        verify_triangles = np.array(verify_triangles)
        # verify_face = mesh.render.render_colors(verify_vertices, verify_triangles, verify_colors, height, width,
        #                                         channel)
        mesh_data = {'vertices': verify_vertices, 'triangles': verify_triangles,
                     'full_triangles': verify_triangles,
                     'colors': verify_colors}
        self.mesh_info = mesh_data
        return self.mesh_info

    def mesh2UVmap(self, mesh_info=None):
        if mesh_info is None:
            assert (self.mesh_info is not None)
            mesh_info = self.mesh_info
        vertices = mesh_info['vertices']
        colors = mesh_info['colors']
        triangles = mesh_info['full_triangles']
        # colors = colors / np.max(colors)
        # model_image = mesh.render.render_colors(vertices, bfm.triangles, colors, image_h, image_w) # only for show

        uv_texture_map = mesh.render.render_colors(self.uv_coords, triangles, colors, self.uv_h, self.uv_w, c=3)
        position = vertices.copy()
        position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
        uv_position_map = mesh.render.render_colors(self.uv_coords, triangles, position, self.uv_h, self.uv_w, c=3)
        self.uv_position_map = uv_position_map
        self.uv_texture_map = uv_texture_map
        return uv_position_map, uv_texture_map

    def setPath(self, image_path, output_dir='data/temp'):
        self.image_path = image_path
        self.image_file_name = image_path.strip().split('/')[-1]
        self.image_name = self.image_file_name.split('.')[0]
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            print('mkdir ', output_dir)
            os.mkdir(output_dir)
        if not os.path.exists(output_dir + '/' + self.image_name):
            os.mkdir(output_dir + '/' + self.image_name)
        self.write_dir = output_dir + '/' + self.image_name

    def readImage(self, image_path=None):
        if image_path is None:
            assert (self.image_path is not None)
            image_path = self.image_path
        self.init_image = io.imread(image_path) / 255.
        self.image_shape = self.init_image.shape

        if not image_path == self.image_path:
            self.image_path = image_path
            self.image_file_name = image_path.strip().split('/')[-1]
            self.image_name = self.image_file_name.split('.')[0]
            if not os.path.exists(self.output_dir + '/' + self.image_name):
                os.mkdir(self.output_dir + '/' + self.image_name)
            self.write_dir = self.output_dir + '/' + self.image_name

    def readBfmFile(self, bfm_path):
        self.bfm_info = sio.loadmat(bfm_path)

    def readMeshFile(self, mesh_path):
        self.mesh_info = sio.loadmat(mesh_path)

    def writeInitImage(self):
        io.imsave(self.write_dir + '/' + self.image_name + '_init.jpg', self.init_image)

    def writeMeshFile(self):
        sio.savemat(self.write_dir + '/' + self.image_name + '_mesh.mat', self.mesh_info)
        if self.is_visualize:
            [height, width, channel] = self.init_image.shape
            mesh_image = mesh.render.render_colors(self.mesh_info['vertices'],
                                                   self.mesh_info['triangles'],
                                                   self.mesh_info['colors'], height, width)
            mesh_image = np.maximum(np.minimum(mesh_image, 1.), 0.)
            io.imsave(self.write_dir + '/' + self.image_name + '_generate.jpg', mesh_image)

    def writeUVmap(self):
        np.save(self.write_dir + '/' + self.image_name + '_uv_posmap.npy', self.uv_position_map)
        if self.uv_texture_map is not None:
            np.save(self.write_dir + '/' + self.image_name + '_uv_texture_map.npy', self.uv_texture_map)
            if self.is_visualize:
                uv_texture_map = np.maximum(np.minimum(self.uv_texture_map, 1.), 0.)
                io.imsave(self.write_dir + '/' + self.image_name + '_uv_texture_map.jpg', uv_texture_map)

    def runPosmap(self):
        # 1. load image and fitted parameters
        [height, width, channel] = self.image_shape
        pose_para = self.bfm_info['Pose_Para'].T.astype(np.float32)
        shape_para = self.bfm_info['Shape_Para'].astype(np.float32)
        exp_para = self.bfm_info['Exp_Para'].astype(np.float32)
        vertices = self.bfm.generate_vertices(shape_para, exp_para)
        # transform mesh

        s = pose_para[-1, 0]
        angles = pose_para[:3, 0]
        t = pose_para[3:6, 0]
        transformed_vertices = self.bfm.transform_3ddfa(vertices, s, angles, t)
        projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
        image_vertices = projected_vertices.copy()
        image_vertices[:, 1] = height - image_vertices[:, 1]

        # 3. crop image with key points
        kpt = image_vertices[self.bfm.kpt_ind, :].astype(np.int32)
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])
        old_bbox = np.array([[left, top], [right, bottom]])

        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        old_size = (right - left + bottom - top) / 2
        size = int(old_size * self.bbox_extend_rate)  # 1.5
        marg = old_size * self.marg_rate  # 0.1
        t_x = np.random.rand() * marg * 2 - marg
        t_y = np.random.rand() * marg * 2 - marg
        center[0] = center[0] + t_x
        center[1] = center[1] + t_y
        size = size * (np.random.rand() * 2 * self.marg_rate - self.marg_rate + 1)

        # crop and record the transform parameters
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        dst_pts = np.array([[0, 0], [0, self.uv_h - 1], [self.uv_w - 1, 0]])
        tform = skimage.transform.estimate_transform('similarity', src_pts, dst_pts)

        # can do some rotations here
        cropped_image = skimage.transform.warp(self.init_image, tform.inverse, output_shape=(self.uv_h, self.uv_w))

        # transform face position(image vertices) along with 2d facial image
        position = image_vertices.copy()
        position[:, 2] = 1
        position = np.dot(position, tform.params.T)
        position[:, 2] = image_vertices[:, 2] * tform.params[0, 0]  # scale z
        position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
        # 4. uv position map: render position in uv space
        uv_position_map = mesh.render.render_colors(self.uv_coords, self.bfm.full_triangles, position, self.uv_h,
                                                    self.uv_w, c=3)

        # get bbox size  2D size as is used in other works
        kpt = position[self.bfm.kpt_ind, :].astype(np.int32)
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])
        bbox = np.array([[left, top], [right, bottom]])

        # get gt landmark68
        init_kpt = self.bfm_info['pt3d_68'].T
        init_kpt[:, 2] = init_kpt[:, 2] - np.min(image_vertices[:, 2])
        new_kpt = copy.copy(init_kpt)
        new_kpt[:, 2] = 1
        new_kpt = np.dot(new_kpt, tform.params.T)
        new_kpt[:, 2] = init_kpt[:, 2] * tform.params[0, 0]

        # 5. save files
        sio.savemat(self.write_dir + '/' + self.image_name + '_bbox_info.mat',
                    {'OldBbox': old_bbox, 'Bbox': bbox, 'Tform': tform.params, 'TformInv': tform._inv_matrix,
                     'Kpt': new_kpt,
                     'OldKpt': init_kpt})
        np.save(self.write_dir + '/' + self.image_name + '_cropped_uv_posmap.npy', uv_position_map)
        io.imsave(self.write_dir + '/' + self.image_name + '_cropped.jpg',
                  (np.squeeze(cropped_image * 255.0)).astype(np.uint8))

        # tex_para = self.bfm_info['Tex_Para'].astype(np.float32)
        # tex = self.bfm.generate_colors(tex_para)
        # self.showMesh({'vertices': position, 'triangles': self.bfm.full_triangles, 'colors': tex})

        # # test rotation

        # angle = np.pi / 2
        # # move-rotate-move
        # t1 = np.array([[1, 0, -128], [0, 1, -128], [0, 0, 1]])
        # r1 = np.array([[math.cos(angle), math.sin(angle), 0], [math.sin(-angle), math.cos(angle), 0], [0, 0, 1]])
        # t2 = np.array([[1, 0, 128], [0, 1, 128], [0, 0, 1]])
        # pos2 = position.copy()
        # pos2[:, 2] = 1
        # rform=t2.dot(r1).dot(t1)
        # pos2 = np.dot(pos2, rform.T)
        # pos2[:, 2] = position[:, 2]
        #
        # transcropped_image = skimage.transform.warp(cropped_image, np.linalg.inv(rform),
        #                                             output_shape=(self.uv_h, self.uv_w))
        # self.showMesh({'vertices': pos2, 'triangles': self.bfm.full_triangles, 'colors': tex})
        #
        # transuvposmap = uv_position_map.copy()
        # for i in range(self.uv_h):
        #     for j in range(self.uv_w):
        #         transuvposmap[i][j][2]=1.
        #         transuvposmap[i][j] = transuvposmap[i][j].dot(rform.T)
        #         transuvposmap[i][j][2]=uv_position_map[i][j][2]
        # uv_tex_map = mesh.render.render_colors(self.uv_coords, self.bfm.full_triangles, tex, self.uv_h,
        #                                        self.uv_w, c=3)
        # self.show([transuvposmap, uv_tex_map, transcropped_image], is_file=False, mode='uvmap')
        # self.show([uv_position_map, uv_tex_map, cropped_image], is_file=False, mode='uvmap')

    def processImage(self, image_path, output_dir):
        self.setPath(image_path, output_dir)
        self.readImage()
        self.readBfmFile(self.image_path.replace('.jpg', '.mat'))
        if self.is_full_image:
            self.bfm2Mesh(self.bfm_info)
            self.mesh2UVmap(self.mesh_info)
            self.writeInitImage()
            self.writeMeshFile()
            self.writeUVmap()
        self.runPosmap()
        self.clear()

    def clear(self):
        self.image_file_name = ''
        self.image_name = ''
        self.image_path = ''
        self.image_dir = ''
        self.output_dir = ''

        self.init_image = None
        self.image_shape = None
        self.bfm_info = None
        self.uv_position_map = None
        self.uv_texture_map = None
        self.mesh_info = None

    @staticmethod
    def showMesh(mesh_info, init_img=None):
        height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
        width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
        channel = 3
        if init_img is not None:
            [height, width, channel] = init_img.shape
        mesh_image = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                               height, width, channel)
        io.imshow(mesh_image)
        plt.show()
        if init_img is not None:
            verify_img = init_img.copy()
            for i in range(height):
                for j in range(width):
                    if (mesh_image[i][j] > 0).any():
                        verify_img[i][j] = mesh_image[i][j]
            io.imshow(verify_img)
            plt.show()
        io.imshow(init_img)
        plt.show()

    def show(self, ipt, is_file=False, mode='image'):
        if mode == 'image':
            if is_file:
                # ipt is a path
                image = io.imread(ipt) / 255.
            else:
                image = ipt
            io.imshow(image)
            plt.show()
        elif mode == 'uvmap':
            # ipt should be [posmap texmap] or [posmap texmap image]
            assert (len(ipt) > 1)
            init_image = None
            if is_file:
                uv_position_map = np.load(ipt[0])
                uv_texture_map = io.imread(ipt[1]) / 255.
                if len(ipt) > 2:
                    init_image = io.imread(ipt[2]) / 255.
            else:
                uv_position_map = ipt[0]
                uv_texture_map = ipt[1]
                if len(ipt) > 2:
                    init_image = ipt[2]
            mesh_info = self.UVmap2Mesh(uv_position_map=uv_position_map, uv_texture_map=uv_texture_map)
            self.showMesh(mesh_info, init_image)
        elif mode == 'mesh':
            if is_file:
                if len(ipt) == 2:
                    mesh_info = sio.loadmat(ipt[0])
                    init_image = io.imread(ipt[1]) / 255.
                else:
                    mesh_info = sio.loadmat(ipt)
                    init_image = None
            else:
                if len(ipt == 2):
                    mesh_info = ipt[0]
                    init_image = ipt[1]
                else:
                    mesh_info = ipt
                    init_image = None
            self.showMesh(mesh_info, init_image)


def workerProcess(image_paths, output_dirs, id, conf):
    print('worker:', id, 'start. task number:', len(image_paths))
    data_processor = DataProcessor(bbox_extend_rate=conf.bboxExtendRate, marg_rate=conf.margin,
                                   is_visualize=conf.isVisualize, is_full_image=conf.isFull)
    for i in range(len(image_paths)):
        # print('\r worker ' + str(id) + ' task ' + str(i) + '/' + str(len(image_paths)) +''+  image_paths[i])
        print("worker {} task {}/{}  {}\r".format(str(id), str(i), str(len(image_paths)), image_paths[i]), end='')
        # output_list[id] = "worker {} task {}/{}  {}".format(str(id), str(i), str(len(image_paths)), image_paths[i])
        data_processor.processImage(image_paths[i], output_dirs[i])
    print('worker:', id, 'end')


def multiProcess(conf):
    worker_num = conf.thread
    input_dir = conf.inputDir
    output_dir = conf.outputDir
    image_path_list = []
    output_dir_list = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for root, dirs, files in os.walk(input_dir):
        temp_output_dir = output_dir
        tokens = root.split(input_dir)
        if not (root.split(input_dir)[1] == ''):
            temp_output_dir = output_dir + root.split(input_dir)[1]
            if not os.path.exists(temp_output_dir):
                os.mkdir(temp_output_dir)

        for file in files:
            file_tokens = file.split('.')
            file_type = file_tokens[1]
            if file_type == 'jpg' or file_type == 'png':
                image_path_list.append(root + '/' + file)
                output_dir_list.append(temp_output_dir)

    total_task = len(image_path_list)
    print('found images:', total_task)

    if worker_num <= 1:
        workerProcess(image_path_list, output_dir_list, 0, conf)
    elif worker_num > 1:
        jobs = []
        task_per_worker = math.ceil(total_task / worker_num)
        st_idx = [task_per_worker * i for i in range(worker_num)]
        ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(worker_num)]
        for i in range(worker_num):
            # temp_data_processor = copy.deepcopy(data_processor)
            p = multiprocessing.Process(target=workerProcess, args=(
                image_path_list[st_idx[i]:ed_idx[i]],
                output_dir_list[st_idx[i]:ed_idx[i]], i, conf))
            jobs.append(p)
            p.start()


if __name__ == "__main__":
    # showModel("data/images/AFLW2000/image00107.jpg", "data/images/AFLW2000/image00107.mat", True)
    # multiProcess("data/images/300W-3D", "data/images/300W-3D-crop", worker_num=8)
    # multiProcess("data/images/AFLW2000", "data/images/AFLW2000-crop", worker_num=2)
    # multiProcess("data/images/300W_LP", "data/images/300W_LP-crop", worker_num=32)
    parser = argparse.ArgumentParser(
        description='data preprocess arguments')

    parser.add_argument('-i', '--inputDir', default='data/images/AFLW2000', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='data/images/AFLW2000-crop', type=str,
                        help='path to the output directory, where results(npy,cropped jpg) will be stored.')
    parser.add_argument('-s', '--isSingle', default=False, type=ast.literal_eval,
                        help='processs one image or all images in a directory')
    parser.add_argument('-t', '--thread', default='1', type=int,
                        help='thread number for multiprocessing')

    parser.add_argument('-f', '--isFull', default=False, type=ast.literal_eval,
                        help='whether to process init image')
    # update in 2017/4/10
    parser.add_argument('-v', '--isVisualize', default=False, type=ast.literal_eval,
                        help='whether to save images of some data such as texture')

    parser.add_argument('-b', '--bboxExtendRate', default=1.5, type=float,
                        help='extend rate of bounding box of cropped face')
    parser.add_argument('-m', '--margin', default=0.1, type=float,
                        help='margin for the bbox')
    conf = parser.parse_args()

    if not conf.isSingle:
        multiProcess(conf)
    else:
        workerProcess([conf.inputDir], [conf.outputDir], 0, conf)
