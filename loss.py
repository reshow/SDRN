import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Conv2DTranspose
from keras.layers import Add, GlobalMaxPooling2D
from keras.layers import add, Flatten
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, Callback, History
from keras.optimizers import adam
from keras import backend as K
from keras.utils import multi_gpu_model
from skimage import io, transform
import os
import matplotlib.pyplot as plt
import math
from data import ImageData, FitGenerator
import time
import argparse
import ast
import scipy.io as sio
import copy
from data import default_init_image_shape, default_cropped_image_shape, default_uvmap_shape, uv_coords, bfm, uv_kpt
from data import face_mask_np, face_mask_mean_fix_rate
from data import bfm2Mesh, mesh2UVmap, UVmap2Mesh, renderMesh

weight_mask = io.imread('uv-data/uv_weight_mask.png') / 255.
weight_mask = K.variable(weight_mask)
face_mask = K.variable(face_mask_np)


def PRNLoss(is_foreface=False, is_weighted=False):
    """
    here is a tricky way to customize loss functions for keras
    :param is_foreface:
    :param is_weighted:
    :return: loss function in keras format
    """

    def templateLoss(y_true, y_pred):
        dist = K.sqrt(K.sum(K.square(K.abs(y_true - y_pred)), axis=-1))
        if is_weighted:
            dist = dist * weight_mask
        if is_foreface:
            dist = dist * face_mask * face_mask_mean_fix_rate
        loss = K.mean(dist)
        return loss

    return templateLoss


def getLossFunction(loss_func_name='SquareError'):
    if loss_func_name == 'RootSquareError' or loss_func_name == 'rse':
        return PRNLoss(is_foreface=False, is_weighted=False)
    elif loss_func_name == 'WeightedRootSquareError' or loss_func_name == 'wrse':
        return PRNLoss(is_foreface=False, is_weighted=True)
    elif loss_func_name == 'ForefaceRootSquareError' or loss_func_name == 'frse':
        return PRNLoss(is_foreface=True, is_weighted=False)
    elif loss_func_name == 'ForefaceWeightedRootSquareError' or loss_func_name == 'fwrse':
        return PRNLoss(is_foreface=True, is_weighted=True)
    else:
        print('unknown loss:', loss_func_name)


def PRNError(is_2d=False, is_normalized=True, is_foreface=True, is_landmark=False, is_gt_landmark=False):
    def templateError(y_true, y_pred, bbox=None, landmarks=None):
        assert (not (is_foreface and is_landmark))
        if is_landmark:
            # the gt landmark is not the same as the landmarks get from mesh using index
            if is_gt_landmark:
                gt = landmarks
                pred = y_pred[uv_kpt[:, 0], uv_kpt[:, 1]]
                diff = np.square(gt - pred)
                if is_2d:
                    dist = np.sqrt(np.sum(diff[:, 0:2], axis=-1))
                else:
                    dist = np.sqrt(np.sum(diff, axis=-1))
            else:
                gt = y_true[uv_kpt[:, 0], uv_kpt[:, 1]]
                pred = y_pred[uv_kpt[:, 0], uv_kpt[:, 1]]
                diff = np.square(gt - pred)
                if is_2d:
                    dist = np.sqrt(np.sum(diff[:, 0:2], axis=-1))
                else:
                    dist = np.sqrt(np.sum(diff, axis=-1))
        else:
            diff = np.square(y_true - y_pred)
            if is_2d:
                dist = np.sqrt(np.sum(diff[:, :, 0:2], axis=-1))
            else:
                # 3d
                dist = np.sqrt(np.sum(diff, axis=-1))
            if is_foreface:
                dist = dist * face_mask_np * face_mask_mean_fix_rate

        if is_normalized:
            # bbox_size = np.sqrt(np.sum(np.square(bbox[0, :] - bbox[1, :])))
            bbox_size = np.sqrt((bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1]))
        else:
            bbox_size = 1.
        loss = np.mean(dist / bbox_size)
        return loss

    return templateError


def getErrorFunction(error_func_name='NME'):
    if error_func_name == 'nme2d' or error_func_name == 'normalized mean error2d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=True)
    elif error_func_name == 'nme3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=True)
    elif error_func_name == 'landmark2d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True)
    elif error_func_name == 'landmark3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True)
    elif error_func_name == 'gtlandmark2d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True)
    elif error_func_name == 'gtlandmark3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True)
    else:
        print('unknown error:', error_func_name)
