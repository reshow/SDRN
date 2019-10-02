import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io
from torchdata import UVmap2Mesh, uv_kpt, bfm2Mesh, getLandmark, mesh2UVmap, bfm
from visualize import showLandmark


def getImageAttentionMask(image, posmap, mode='hard'):
    """
    需要加一个正态分布吗？
    """
    [image_h, image_w, image_c] = image.shape()
    kpt = getLandmark(posmap)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    showLandmark(image, kpt)

    mask = np.zeros((image_h, image_w)).astype(np.float32)
    mask[left:right, top:bottom] = 1.0
