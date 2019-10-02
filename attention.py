import numpy as np
import numba
from torchdata import UVmap2Mesh, uv_kpt, bfm2Mesh, getLandmark, mesh2UVmap, bfm, face_mask_np
from faceutil import mesh
import cv2

face_mask_np3d = np.stack([face_mask_np, face_mask_np, face_mask_np], axis=2)


def getImageAttentionMask(image, posmap, mode='hard'):
    """
    需要加一个正态分布吗？
    """
    [height, width, channel] = image.shape
    p = (posmap * face_mask_np3d).clip(0, 255).astype(int)
    mask = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         [x, y, z] = posmap[i, j]
    #         x = int(x)
    #         y = int(y)
    #         mask[y, x] = 1
    mask[p[:, :, 1], p[:, :, 0]] = 1

    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.ceil(np.array(blur)).astype(np.uint8)
    return mask
