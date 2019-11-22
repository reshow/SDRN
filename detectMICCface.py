import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, warp
import os
from PIL import Image
import scipy.io as sio


def addBlankData(path='data/images/micc'):
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'init.jpg' in str(file):
                print(file)
                image_dir = root
                image_name = file.split('_init.jpg')[0]

                image = io.imread(image_dir + '/' + image_name + '_cropped.jpg')

                pos = np.load(image_dir + '/' + image_name + '_mesh.npy')
                np.save(image_dir + '/' + image_name + '_mesh.npy', pos)

                np.save(image_dir + '/' + image_name + '_cropped_uv_posmap.npy', np.random.rand(256, 256, 3))
                cropped_image_path = image_dir + '/' + image_name + '_cropped.npy'
                bbox_info_path = image_dir + '/' + image_name + '_bbox_info.mat'
                offset_posmap_path = image_dir + '/' + image_name + '_offset_posmap.npy'
                attention_mask_path = image_dir + '/' + image_name + '_attention_mask.npy'

                np.save(cropped_image_path, image)
                np.save(offset_posmap_path, np.random.rand(256, 256, 3))
                fake_mask = np.random.rand(256, 256)
                fake_mask[fake_mask > 0.5] = 1
                fake_mask[fake_mask <= 0.5] = 0
                np.save(attention_mask_path, fake_mask.astype(np.uint8))
                sio.savemat(bbox_info_path, {'OldBbox': np.array([[1, 2], [3, 4]]), 'Bbox': np.array([[1, 2], [3, 4]]), 'Tform': np.random.rand(3, 3),
                                             'TformInv': np.random.rand(3, 3),
                                             'Tform3d': np.random.rand(4, 4), 'Kpt': np.random.rand(68, 3), 'OldKpt': np.random.rand(68, 3),
                                             'TformOffset': np.random.rand(4, 4)})



# def cropMICC():
#     import dlib
#     detector_path = 'E:/face/FR&DA/PRNet/Data/net-data/mmod_human_face_detector.dat'
#     predictor_path = 'E:/face/Wang Demo/data/shape_predictor_68_face_landmarks.dat'
#     face_detector = dlib.cnn_face_detection_model_v1(
#         detector_path)
#     predictor = dlib.shape_predictor(predictor_path)
#     resolution_inp = 256
#
#     for root, dirs, files in os.walk('data/images/micc'):
#         for file in files:
#             if 'init.jpg' in str(file):
#                 image = io.imread(root + '/' + str(file))
#                 pos = np.load(root + '/' + str(file).replace('init.jpg', 'initpos.npy'))
#
#                 d_img = Image.fromarray(image)
#                 d_img = d_img.resize((75, 75), Image.BILINEAR)
#                 d_img = np.array(d_img)
#                 dets = face_detector(d_img, 1)
#                 if len(dets) > 0:
#                     d = dets[0].rect
#                     left = d.left() * 8
#                     right = d.right() * 8
#                     top = d.top() * 8
#                     bottom = d.bottom() * 8
#                     old_size = (right - left + bottom - top) / 2
#                 else:
#                     left = pos[:, 0].min()
#                     right = pos[:, 0].max()
#                     top = pos[:, 1].min()
#                     bottom = pos[:, 1].max() - 20
#                     old_size = (right - left + bottom - top) / 2.75
#
#                 center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
#                 size = int(old_size * 1.58)
#                 src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
#                                     [center[0] + size / 2, center[1] - size / 2]])
#                 DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
#                 tform = estimate_transform('similarity', src_pts, DST_PTS)
#                 trans_mat = tform.params
#                 trans_mat_inv = tform._inv_matrix
#                 scale = trans_mat[0][0]
#
#                 position = pos.copy()
#                 position[:, 2] = 1
#                 position = np.dot(position, trans_mat.T)
#                 position[:, 2] = pos[:, 2] * scale  # scale z
#                 position[:, 2] = position[:, 2] - np.min(position[:, 2])
#
#                 cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
#                 io.imsave(root + '/' + str(file).replace('init.jpg', 'cropped.jpg'), cropped_image)
#                 np.save(root + '/' + str(file).replace('init.jpg', 'mesh.npy'), position)
#                 print(str(file))


def singleProcess(path):
    resolution_inp = 256

    root = path
    file = path.split('/')[-1] + '_init.jpg'

    image = io.imread(root + '/' + str(file))
    pos = np.load(root + '/' + str(file).replace('init.jpg', 'initpos.npy'))

    left = pos[:, 0].min()
    right = pos[:, 0].max()
    top = pos[:, 1].min()
    bottom = pos[:, 1].max() - 20
    old_size = (right - left + bottom - top) / 2.5

    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 - old_size * 0.1])
    size = int(old_size * 1.58)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    trans_mat = tform.params
    trans_mat_inv = tform._inv_matrix
    scale = trans_mat[0][0]

    position = pos.copy()
    position[:, 2] = 1
    position = np.dot(position, trans_mat.T)
    position[:, 2] = pos[:, 2] * scale  # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2])

    cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
    io.imsave(root + '/' + str(file).replace('init.jpg', 'cropped.jpg'), cropped_image)
    np.save(root + '/' + str(file).replace('init.jpg', 'mesh.npy'), position)
    print(str(file))


size_list = [[95, 425], [75, 375], [45, 405], [90, 430], [75, 475], [95, 445], [35, 415], [115, 445], [90, 450], [20, 450],
             [80, 450], [50, 410], [100, 440], [100, 420], [50, 410], [100, 430], [80, 420], [95, 435], [95, 405], [80, 430],
             [110, 390], [95, 450], [80, 410], [100, 430], [70, 400], [50, 400], [80, 450], [80, 460], [50, 380], [100, 420],
             [70, 400], [70, 450], [80, 430], [80, 440], [90, 450], [70, 470], [90, 420], [90, 380], [100, 470], [80, 430],
             [100, 380], [60, 420], [70, 410], [50, 440], [80, 410], [150, 460], [100, 480], [70, 390], [20, 440], [100, 450]]


def cropMICC2():
    resolution_inp = 256

    for root, dirs, files in os.walk('data/images/micc'):
        for file in files:
            if 'init.jpg' in str(file):
                image = io.imread(root + '/' + str(file))
                pos = np.load(root + '/' + str(file).replace('init.jpg', 'initpos.npy'))

                image_id = int(str(file).split('_')[0]) // 10000 - 1
                up = size_list[image_id][0]
                down = size_list[image_id][1]

                center = np.array([300, up + (down - up) / 2.])
                size = (down - up) * 0.67 * 1.75

                # center = np.array([300, 275])
                # size = 550

                src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                                    [center[0] + size / 2, center[1] - size / 2]])
                DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
                tform = estimate_transform('similarity', src_pts, DST_PTS)
                trans_mat = tform.params
                trans_mat_inv = tform._inv_matrix
                scale = trans_mat[0][0]

                position = pos.copy()
                position[:, 2] = 1
                position = np.dot(position, trans_mat.T)
                position[:, 2] = pos[:, 2] * scale  # scale z
                position[:, 2] = position[:, 2] - np.min(position[:, 2])

                cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
                io.imsave(root + '/' + str(file).replace('init.jpg', 'cropped.jpg'), cropped_image)
                np.save(root + '/' + str(file).replace('init.jpg', 'mesh.npy'), position)
                print(str(file))


# singleProcess()
# cropMICC2()
addBlankData()
