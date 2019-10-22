import data
import numpy as np
import os
import scipy.io as sio
from skimage import io, transform
import skimage
from faceutil import mesh
import argparse
import ast
import copy
import multiprocessing
import math
from data import default_init_image_shape, default_cropped_image_shape, default_uvmap_shape, uv_coords, bfm
from data import face_mask_np, face_mask_mean_fix_rate
from data import bfm2Mesh, mesh2UVmap, UVmap2Mesh, renderMesh, getTransformMatrix
from augmentation import getRotateMatrix, getRotateMatrix3D
from numpy.linalg import inv
from masks import getImageAttentionMask, getVisibilityMask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='data preprocess arguments')

    parser.add_argument('-o', '--outputDir', default='data/images/Extra-LP', type=str,
                        help='path to the output directory, where results(npy,cropped jpg) will be stored.')
    conf = parser.parse_args()
    if not os.path.exists(conf.outputDir):
        os.mkdir(conf.outputDir)
