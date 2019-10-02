import numpy as np
import numba
from torchdata import UVmap2Mesh, uv_kpt, bfm2Mesh, getLandmark, mesh2UVmap, bfm, face_mask_np
from faceutil import mesh


@numba.jit
def getImageAttentionMask(image, posmap, mode='hard'):
    """
    需要加一个正态分布吗？
    """
    [height, width, channel] = image.shape
    mask = np.zeros((height, width)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            [x, y, z] = posmap[i, j]
            x = int(x)
            y = int(y)
            mask[y, x] = 1

    return mask.astype(np.uint8)
