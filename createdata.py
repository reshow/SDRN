import numpy as np
import os
import scipy.io as sio
from skimage import io, transform
import skimage
from faceutil import mesh
import argparse
from data import default_init_image_shape, default_cropped_image_shape, default_uvmap_shape, uv_coords, bfm
from data import face_mask_np, face_mask_mean_fix_rate
from data import bfm2Mesh, mesh2UVmap, UVmap2Mesh, renderMesh, getTransformMatrix
from numpy.linalg import inv
from masks import getImageAttentionMask, getVisibilityMask
from processor import DataProcessor
from augmentation import randomErase
import numba

data_processor = DataProcessor()


def addModuleData(data_dir):
    module_path_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_tokens = file.split('.')
            file_type = file_tokens[1]
            if file_type == 'jpg':
                module_path_list.append(root + '/' + str(file).replace('jpg', 'mat'))
    print('add %d images' % len(module_path_list))
    return module_path_list


def avoidOutOfImage(pose_para, shape_para, exp_para, height=450):
    vertices = bfm.generate_vertices(shape_para, exp_para)
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]

    T_bfm = getTransformMatrix(s, angles, t, height)
    temp_ones_vec = np.ones((len(vertices), 1))
    vertices_4dim = np.concatenate((vertices, temp_ones_vec), axis=-1)
    image_vertices = vertices_4dim.dot(T_bfm.T)[:, 0:3]
    dx = 0
    dy = 0
    if image_vertices[:, 0].min() < 0:
        dx = -image_vertices[:, 0].min()
    if image_vertices[:, 0].max() > 450:
        dx = -(image_vertices[:, 0].max() - 450)
    if image_vertices[:, 1].min() < 0:
        dy = -image_vertices[:, 1].min()
    if image_vertices[:, 1].max() > 450:
        dy = -(image_vertices[:, 1].max() - 450)
    pose_para[3][0] += dx
    pose_para[4][0] += dy
    return pose_para


@numba.njit
def numbaCross3D(a, b):
    # c = np.zeros(3)
    # c[0] = a[1] * b[2] - b[1] * a[2]
    # c[1] = a[2] * b[0] - b[2] * a[0]
    # c[2] = a[0] * b[1] - b[0] * a[1]
    return a[0] * b[1] - b[0] * a[1]


@numba.jit
def filterTriangles(new_triangles, vertices, triangles):
    for i in range(len(triangles)):
        t = triangles[i]
        a = vertices[int(t[0])]
        b = vertices[int(t[1])]
        c = vertices[int(t[2])]
        direct = numbaCross3D(b - a, c - b)
        if direct >= 0:
            new_triangles[i] = t
    return new_triangles


def renderTriangleFix(mesh_info):
    vertices = mesh_info['vertices']
    triangles = mesh_info['triangles']
    new_triangles = np.zeros((len(triangles), 3))
    new_triangles = filterTriangles(new_triangles, vertices, triangles)
    return np.array(new_triangles)


def createImageData(image_name, root, tex_modules, pos_modules):
    write_dir = root + '/' + image_name
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    margin = 0.1
    tex_module_id = np.random.randint(0, len(tex_modules))
    tex_mat = sio.loadmat(tex_modules[tex_module_id])
    tex_para = tex_mat['Tex_Para'].astype(np.float32)
    color_para = tex_mat['Color_Para'].astype(np.float32)
    illum_para = tex_mat['Illum_Para'].astype(np.float32)

    for i in range(len(tex_para)):
        rate = 1 - margin + np.random.rand() * margin * 2
        tex_para[i] = tex_para[i] * rate
    for i in range(len(color_para)):
        rate = 1 - margin + np.random.rand() * margin * 2
        color_para[i] = color_para[i] * rate
    for i in range(len(illum_para)):
        rate = 1 - margin + np.random.rand() * margin * 2
        illum_para[i] = illum_para[i] * rate

    pos_module_id = np.random.randint(0, len(pos_modules))
    pos_mat = sio.loadmat(pos_modules[pos_module_id])

    pose_para = pos_mat['Pose_Para'].T.astype(np.float32)
    shape_para = pos_mat['Shape_Para'].astype(np.float32)
    exp_para = pos_mat['Exp_Para'].astype(np.float32)

    for i in range(len(pose_para)):
        rate = 1 - margin + np.random.rand() * margin * 2
        pose_para[i] = pose_para[i] * rate
    pose_para[0][0] = (np.random.randint(0, 2) * 2 - 1) * np.pi / 9 * np.random.rand()
    pose_para[1][0] = (np.random.randint(0, 2) * 2 - 1) * np.pi / 2 * (1 - margin + np.random.rand() * margin * 2)
    pose_para[2][0] = (np.random.randint(0, 2) * 2 - 1) * np.pi / 9 * np.random.rand()

    for i in range(len(shape_para)):
        rate = 1 - margin + np.random.rand() * margin * 2
        shape_para[i] = shape_para[i] * rate
    for i in range(len(exp_para)):
        rate = 1 - margin + np.random.rand() * margin * 2
        exp_para[i] = exp_para[i] * rate

    pose_para = avoidOutOfImage(pose_para, shape_para, exp_para, height=450)

    init_image = np.ones((450, 450, 3))
    # background color
    init_image = (init_image * np.random.rand(3)).astype(np.float32)
    init_image = randomErase(init_image, 50, )

    bfm_info = {'Tex_Para': tex_para, 'Color_Para': color_para, 'Illum_Para': illum_para, 'Pose_Para': pose_para.T, 'Shape_Para': shape_para,
                'Exp_Para': exp_para}
    mesh_info = bfm2Mesh(bfm_info)
    new_triangles = renderTriangleFix(mesh_info)
    # new_triangles=mesh_info['triangles']
    mesh_image = mesh.render.render_colors(mesh_info['vertices'],
                                           new_triangles,  # mesh_info['triangles'],
                                           mesh_info['colors'], 450, 450, BG=init_image)
    mask_colors = np.ones((len(mesh_info['colors']), 3))
    mask_image = mesh.render.render_colors(mesh_info['vertices'],
                                           new_triangles,  # mesh_info['triangles'],
                                           mask_colors, 450, 450)[:, :, 0]
    mesh_image = np.clip(mesh_image, 0., 1.)

    # 1. start
    [height, _, _] = init_image.shape
    vertices = bfm.generate_vertices(shape_para, exp_para)
    offset_vertices = bfm.generate_offset(shape_para, exp_para)

    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]

    T_bfm = getTransformMatrix(s, angles, t, height)
    temp_ones_vec = np.ones((len(vertices), 1))
    vertices_4dim = np.concatenate((vertices, temp_ones_vec), axis=-1)
    image_vertices = vertices_4dim.dot(T_bfm.T)[:, 0:3]

    # 3. crop image with key points
    # 3.1 get old bbox
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    [left, top, right, bottom] = data_processor.getBbox(kpt)
    old_bbox = np.array([[left, top], [right, bottom]])

    # 3.2 add margin to bbox
    [center, size] = data_processor.getCropBox([left, top, right, bottom])

    # 3.3 crop and record the transform parameters
    [crop_h, crop_w, _] = default_cropped_image_shape

    T_3d = np.zeros((4, 4))
    T_3d[0, 0] = crop_w / size
    T_3d[1, 1] = crop_h / size
    T_3d[2, 2] = crop_w / size
    T_3d[3, 3] = 1.

    T_3d[0:3, 3] = [(size / 2 - center[0]) * crop_w / size, (size / 2 - center[1]) * crop_h / size, -np.min(image_vertices[:, 2]) * crop_w / size]
    T_2d = np.zeros((3, 3))
    T_2d[0:2, 0:2] = T_3d[0:2, 0:2]
    T_2d[2, 2] = 1.
    T_2d[0:2, 2] = T_3d[0:2, 3]
    T_2d_inv = inv(T_2d)
    cropped_image = skimage.transform.warp(mesh_image, T_2d_inv, output_shape=(crop_h, crop_w))
    cropped_mask = skimage.transform.warp(mask_image, T_2d_inv, output_shape=(crop_h, crop_w))
    # 3.4 transform face position(image vertices)

    p4d = np.concatenate((image_vertices, temp_ones_vec), axis=-1)
    position = p4d.dot(T_3d.T)[:, 0:3]

    offset_position = offset_vertices * 1e-4
    # T_scale_1e4 = np.diagflat([1e4, 1e4, 1e4, 1])
    # mean_position = bfm.get_mean_shape()
    # rebuild_position = np.concatenate((mean_position * 1e-4 + offset_position, temp_ones_vec), axis=-1).dot(T_3d.dot(T_bfm).dot(T_scale_1e4).T)[:, 0:3]
    # diff = rebuild_position - position

    # 4. uv position map: render position in uv space
    [uv_h, uv_w, uv_c] = default_uvmap_shape
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h,
                                                uv_w, uv_c)

    uv_offset_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, offset_position, uv_h,
                                              uv_w, uv_c)

    # get new bbox
    kpt = position[bfm.kpt_ind, :].astype(np.int32)
    [left, top, right, bottom] = DataProcessor.getBbox(kpt)
    bbox = np.array([[left, top], [right, bottom]])

    # get gt landmark68
    # init_kpt = self.bfm_info['pt3d_68'].T
    init_kpt = image_vertices[bfm.kpt_ind, :]
    init_kpt_4d = np.concatenate((init_kpt, np.ones((68, 1))), axis=-1)
    new_kpt = init_kpt_4d.dot(T_3d.T)[:, 0:3]

    # 5. save files
    # is_augment = True
    # if is_augment:
    #     # attention_mask = getImageAttentionMask(cropped_image, uv_position_map)
    #     # visibility_mask = getVisibilityMask(uv_position_map, cropped_image.shape)
    #     np.save(write_dir + '/' + image_name + '_attention_mask.npy', np.around(cropped_mask[:, :, 0]).astype(np.uint8))
    #     # np.save(write_dir + '/' + image_name + '_visibility_mask.npy', visibility_mask.astype(np.uint8))
    #
    # sio.savemat(write_dir + '/' + image_name + '_bbox_info.mat',
    #             {'OldBbox': old_bbox, 'Bbox': bbox, 'Tform': T_2d.astype(np.float32), 'TformInv': T_2d_inv.astype(np.float32),
    #              'Tform3d': T_3d.astype(np.float32), 'Kpt': new_kpt, 'OldKpt': init_kpt,
    #              'TformOffset': T_3d.dot(T_bfm).astype(np.float32)})
    # np.save(write_dir + '/' + image_name + '_cropped_uv_posmap.npy', uv_position_map.astype(np.float32))
    # np.save(write_dir + '/' + image_name + '_offset_posmap.npy', uv_offset_map.astype(np.float32))
    # io.imsave(write_dir + '/' + image_name + '_cropped.jpg', (np.squeeze(cropped_image * 255.0)).astype(np.uint8))
    # io.imsave(write_dir + '/' + image_name + '_init.jpg', (mesh_image * 255.0).astype(np.uint8))
    # np.save(write_dir + '/' + image_name + '_cropped.npy', (np.squeeze(cropped_image * 255.0)).astype(np.uint8))

    output_prefix = write_dir
    offset_map_path = output_prefix + '/offset_map.npy'
    position_map_path = output_prefix + '/position_map.npy'
    cropped_image_path = output_prefix + '/image.npy'
    visual_image_path = output_prefix + '/image.jpg'
    attention_path = output_prefix + '/attention.jpg'

    io.imsave(attention_path, cropped_mask.astype(np.uint8) * 255)
    np.save(position_map_path, uv_position_map.astype(np.float32))
    np.save(offset_map_path, uv_offset_map.astype(np.float32))
    io.imsave(visual_image_path, (np.squeeze(cropped_image * 255.0)).astype(np.uint8))
    np.save(cropped_image_path, (np.squeeze(cropped_image * 255.0)).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='data preprocess arguments')
    parser.add_argument('-t', '--inputTexDir', default='data/images/300W-3D', type=str,
                        help='path to the output directory, where results(npy,cropped jpg) will be stored.')
    parser.add_argument('-p', '--inputPosDir', default='data/images/AFLW2000', type=str,
                        help='path to the output directory, where results(npy,cropped jpg) will be stored.')
    parser.add_argument('-o', '--outputDir', default='data/images/Extra-LP', type=str,
                        help='path to the output directory, where results(npy,cropped jpg) will be stored.')
    parser.add_argument('-n', '--num', default=2000, type=int)
    conf = parser.parse_args()
    if not os.path.exists(conf.outputDir):
        os.mkdir(conf.outputDir)

    tex_modules = addModuleData(conf.inputTexDir)
    pos_modules = addModuleData(conf.inputPosDir)
    for i in range(conf.num):
        print('\r', i, end='')
        createImageData('image' + '%.4d' % i, conf.outputDir, tex_modules, pos_modules)
