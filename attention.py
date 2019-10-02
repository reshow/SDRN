import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io
from torchdata import UVmap2Mesh, uv_kpt, bfm2Mesh, getLandmark, mesh2UVmap, bfm, face_mask_np
from visualize import showLandmark
from faceutil import mesh


def getImageAttentionMask(image, posmap, mode='hard'):
    """
    需要加一个正态分布吗？
    """
    [height, width, channel] = image.shape
    tex = np.zeros((height, width, channel))
    tex[:, :, :] = 2

    mesh_info = UVmap2Mesh(posmap, tex, True)
    mesh_image = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                           height, width, channel)
    mask = np.clip(mesh_image, 0, 1).astype(np.float32)
    return mask
