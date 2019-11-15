import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
from math import cos, sin
import skimage.io as io

IMAGE_WIDTH = 600
r = pyrender.OffscreenRenderer(IMAGE_WIDTH, IMAGE_WIDTH)
scene = pyrender.Scene()


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


def example(path='data/back/MICC/subject_01/Model/frontal1/obj/110920150452.obj'):
    scene.clear()
    objmesh = trimesh.load(path)
    mesh = pyrender.Mesh.from_trimesh(objmesh)
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # camera = pyrender.OrthographicCamera(xmag=150, ymag=150, zfar=1000)
    camera_pose = np.array([
        [1, 0, 0, 30],
        [0, 1, 0.0, 0.0],
        [0.0, 0, 1, 250],
        [0.0, 0.0, 0.0, 1.0], ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=camera_pose)
    # light = pyrender.SpotLight(color=np.ones(3), intensity=30,
    #                             innerConeAngle=np.pi/16.0,
    #                             outerConeAngle=np.pi/6.0)
    # scene.add(light, pose=camera_pose)

    color, depth = r.render(scene)
    plt.imshow(color)
    plt.show()


def tryid(i):
    data_dir = 'data/back/MICC' + '/' + 'subject_%.2d' + '/Model/frontal1/obj' % i
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.split('.')[-1] == 'obj':
                example(root + '/' + file)


def listdir(root_dir):
    otp = open('data/miccfilelist.txt', 'w', encoding='utf-8')
    for i in range(1, 10):
        data_dir = root_dir + '/' + 'subject_0' + str(i) + '/Model/frontal1/obj'
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.split('.')[-1] == 'obj':
                    otp.write(root + '/' + file + '\n')
    for i in range(10, 54):
        data_dir = root_dir + '/' + 'subject_' + str(i) + '/Model/frontal1/obj'
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.split('.')[-1] == 'obj':
                    otp.write(root + '/' + file + '\n')
    otp.close()


def readFileList(txtpath='data/miccfilelist.txt'):
    ipt = open(txtpath, 'r', encoding='utf-8')
    file_list = []
    for line in ipt.readlines():
        file_list.append(line.strip())
    return file_list


def renderObj(obj, x, y):
    mesh = pyrender.Mesh.from_trimesh(obj)
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    pitch_angle = x / 180. * np.pi
    yaw_angle = y / 180. * np.pi

    R2 = angle2Matrix(0, yaw_angle, 0)
    R1 = angle2Matrix(np.pi / 6. + pitch_angle, 0, 0)

    R = R2.dot(R1)
    face_pose = np.zeros((4, 4))
    face_pose[0:3, 0:3] = R
    face_pose[3, 3] = 1
    scene.add(mesh, pose=face_pose)

    camera_pose = np.eye(4)
    camera_pose[2, 3] = (obj.vertices[:, 1].max() - obj.vertices[:, 1].min()) / 1.5

    camera = pyrender.OrthographicCamera(xmag=camera_pose[2, 3], ymag=camera_pose[2, 3], zfar=1000)

    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=50.0)
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)
    # plt.imshow(color)
    # plt.show()
    scene.clear()

    v = mesh.primitives[0].positions
    p = v.copy()
    pt = p.dot(R.T)

    pos = v.copy()

    for i in range(len(pt)):
        pos[i][0] = pt[i][0] * IMAGE_WIDTH / 2.0 / camera_pose[2, 3] + IMAGE_WIDTH / 2.0
        pos[i][1] = IMAGE_WIDTH / 2.0 - pt[i][1] * IMAGE_WIDTH / 2.0 / camera_pose[2, 3]
        pos[i][2] = pt[i][2] * IMAGE_WIDTH / 2.0 / camera_pose[2, 3]

    return color, pos


def renderObjList(obj, target_dir, id):
    obj.vertices = obj.vertices - obj.vertices.mean(axis=0)
    for y in range(-80, 81, 40):
        for x in [0, -15, 20, 25]:
            image, pos = renderObj(obj, x, y)
            save_folder = target_dir + '/' + str(id)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            io.imsave(save_folder + '/' + str(id) + '_init.jpg', image)
            np.save(save_folder + '/' + str(id) + '_initpos.npy', pos)
            id = id + 1


def process(file_list, target_dir):
    rendered_num = 10000
    for file in file_list:
        scene.clear()
        objmesh = trimesh.load(file)
        print(file)
        renderObjList(objmesh, target_dir, rendered_num)

        rendered_num = rendered_num + 10000


if __name__ == '__main__':
    input_dir = 'data/back/MICC'
    output_dir = 'data/images/micc'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_list = readFileList()
    process(file_list, output_dir)
