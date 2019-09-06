import os
import sys
import numpy as np
import scipy.io as sio
from skimage import io
from faceutil import mesh
# from data import bfm, modelParam2Mesh, UVMap2Mesh
import matplotlib.pyplot as plt
from skimage import io, transform
# from test import readUVKpt
import open3d as o3d



def showMesh(image_path, mat_path, is_show_in_init_image=False):
    image_name = image_path.strip().split('/')[-1]
    image = io.imread(image_path) / 255.
    [h, w, c] = image.shape

    # --load mesh data
    C = sio.loadmat(mat_path)
    vertices = C['vertices']
    colors = C['colors']
    triangles = C['full_triangles']
    colors = colors / np.max(colors)
    mesh_image = mesh.render.render_colors(vertices, bfm.triangles, colors, h, w)
    # io.imsave('{}/initmodel.jpg'.format(output_dir), modelimage / np.max(modelimage))
    mesh_image = mesh_image / np.max(mesh_image)
    if is_show_in_init_image:
        verify_map = image.copy()
        # paste onto the init image
        for i in range(h):
            for j in range(w):
                if (mesh_image[i][j] > 0).any():
                    verify_map[i][j] = mesh_image[i][j]
        io.imshow(verify_map)
        plt.show()
        io.imshow(mesh_image)
        plt.show()
        io.imshow(image)
        plt.show()
    else:
        io.imshow(mesh_image)
        plt.show()


def showModel(image_path, mat_path, is_show_in_init_image=False):
    # 1. load image and fitted parameters
    image = io.imread(image_path) / 255.
    [image_h, image_w, channel] = image.shape
    print(image.shape)
    info = sio.loadmat(mat_path)
    [image_vertices, tex_color] = modelParam2Mesh(info, image.shape)

    # render
    model_image = mesh.render.render_colors(image_vertices, bfm.triangles, tex_color, image_h, image_w)
    # model_image = model_image / np.max(model_image)

    if is_show_in_init_image:
        verify_map = image.copy()
        # paste onto the init image
        for i in range(image_h):
            for j in range(image_w):
                if (model_image[i][j] > 0).any():
                    verify_map[i][j] = model_image[i][j]
        io.imshow(verify_map)
        plt.show()
        io.imshow(model_image)
        plt.show()
        io.imshow(image)
        plt.show()
    else:
        io.imshow(model_image)
        plt.show()


def showUVMap(pos_map_path, tex_map_path, image_path, is_show_in_init_image=False):
    pos_map = np.load(pos_map_path)
    if tex_map_path is not None:
        tex_map = io.imread(tex_map_path) / 255.
    else:
        red = np.ones((256, 256))
        red = red / 3
        tex_map = np.zeros((256, 256, 3))
        tex_map[:, :, 0] = red
    image = io.imread(image_path) / 255.
    [h, w, c] = image.shape
    [vertices, colors, triangles, mesh_image] = UVMap2Mesh(pos_map, tex_map, image.shape)
    if is_show_in_init_image:
        verify_map = image.copy()
        # paste onto the init image
        for i in range(h):
            for j in range(w):
                if (mesh_image[i][j] > 0).any():
                    verify_map[i][j] = mesh_image[i][j]
        io.imshow(verify_map)
        plt.show()
        io.imshow(mesh_image)
        plt.show()
        io.imshow(image)
        plt.show()
    else:
        io.imshow(mesh_image)
        plt.show()

    # # open3d mesh renderer
    # temp_mesh=o3d.geometry.TriangleMesh()
    #
    # temp_mesh.vertices.extend(vertices)
    #
    # temp_mesh.triangles.extend(triangles)
    # temp_mesh.paint_uniform_color([0.2,0.5,1])
    # temp_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([temp_mesh])
    # return temp_mesh


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


def showKpt(file_path, is_show_init=False):
    uv_kpt = readUVKpt('uv-data/uv_kpt_ind.txt')

    image_name = file_path.split('/')[-1]
    image_path = file_path + '/' + image_name + '_cropped.jpg'
    crop_image = io.imread(image_path) / 255.
    crop_image = transform.resize(crop_image, (256, 256, 3))
    gt_path = file_path + '/' + image_name + '_cropped_uv_posmap.npy'
    gt = np.load(gt_path)

    image = crop_image.copy()
    kpt = gt[uv_kpt[:, 0], uv_kpt[:, 1]]
    kpt = np.round(kpt).astype(int)
    # weak perspective projection so that the xy of 3D coordinate is exactly the 2D coordinate
    image[kpt[:, 1], kpt[:, 0]] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] - 1] = np.array([1, 0, 0])

    io.imshow(image)
    plt.show()

    if is_show_init:
        image_path = file_path + '/' + image_name + '_init.jpg'
        init_image = io.imread(image_path) / 255.
        gt_path = file_path + '/' + image_name + '_uv_posmap.npy'
        gt = np.load(gt_path)

        image = init_image.copy()
        kpt = gt[uv_kpt[:, 0], uv_kpt[:, 1]]
        kpt = np.round(kpt).astype(int)
        # weak perspective projection so that the xy of 3D coordinate is exactly the 2D coordinate
        image[kpt[:, 1], kpt[:, 0]] = np.array([1, 0, 0])
        image[kpt[:, 1] + 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
        image[kpt[:, 1] - 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
        image[kpt[:, 1] - 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
        image[kpt[:, 1] + 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
        io.imshow(image)
        plt.show()


def showimg(input, is_path=True):
    if is_path:
        img = io.imread(input) / 255.
        io.imshow(img)
        plt.show()
    else:
        io.imshow(input)
        plt.show()


if __name__ == "__main__":
    showKpt('data/images/300W-3D-crop/AFW/134212_1', is_show_init=True)
    # showUVMap('data/images/AFLW2000-out/image00002/image00002_uv_posmap.npy', None,
    #           # 'data/images/AFLW2000-output/image00002/image00002_uv_texture_map.jpg',
    #           'data/images/AFLW2000-out/image00002/image00002_init.jpg', True)

