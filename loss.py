import numpy as np
from skimage import io, transform
from data import uv_kpt
from data import face_mask_np, face_mask_mean_fix_rate, getWeightedKpt
from dataloader import toTensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from icp import icp

weight_mask_np = io.imread('uv-data/uv_weight_mask.png').astype(float)
weight_mask_np[weight_mask_np == 255] = 256
weight_mask_np = weight_mask_np / 16

weight_mask = torch.from_numpy(weight_mask_np)
face_mask = torch.from_numpy(face_mask_np)
face_mask_3D = toTensor(np.repeat(np.reshape(face_mask_np, (256, 256, 1)), 3, -1))
if torch.cuda.is_available():
    weight_mask = weight_mask.cuda().float()
    face_mask = face_mask.cuda().float()
    face_mask_3D = face_mask_3D.cuda().float()


def UVLoss(is_foreface=False, is_weighted=False, is_nme=False):
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate
            self.weight_mask = nn.Parameter(weight_mask.clone())
            self.face_mask = nn.Parameter(face_mask.clone())
            self.weight_mask.requires_grad = False
            self.face_mask.requires_grad = False

        def forward(self, y_true, y_pred):
            if is_nme:
                y_true = y_true * self.face_mask
                y_pred = y_pred * self.face_mask
                y_true[:, 2, :, :] = y_true[:, 2, :, :] - torch.mean(y_true[:, 2, :, :]) * face_mask_mean_fix_rate
                y_pred[:, 2, :, :] = y_pred[:, 2, :, :] - torch.mean(y_pred[:, 2, :, :]) * face_mask_mean_fix_rate
            dist = torch.sqrt(torch.sum((y_true - y_pred) ** 2, 1))
            if is_weighted:
                dist = dist * self.weight_mask
            if is_foreface:
                dist = dist * (self.face_mask * face_mask_mean_fix_rate)
            if is_nme:
                yt = y_true.permute(0, 2, 3, 1)
                kpt = yt[:, uv_kpt[:, 0], uv_kpt[:, 1]]
                left = torch.min(kpt[:, :, 0], dim=1)[0]
                right = torch.max(kpt[:, :, 0], dim=1)[0]
                top = torch.min(kpt[:, :, 1], dim=1)[0]
                bottom = torch.max(kpt[:, :, 1], dim=1)[0]
                bbox_size = torch.sqrt((right - left) * (bottom - top))
                # dist = torch.mean(dist, dim=(1, 2))
                dist = dist / bbox_size

            loss = torch.mean(dist)
            return loss * self.rate

    return TemplateLoss


def ParamLoss(mode):
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate
            self.mode = mode

        def forward(self, y_true, y_pred):
            if self.mode == 'mae':
                dist = torch.mean(torch.abs(y_true - y_pred))
            elif self.mode == 'mse':
                dist = F.mse_loss(y_pred, y_true)
            elif self.mode == 'rmse':
                dist = torch.mean(torch.sqrt(torch.sum((y_true - y_pred) ** 2, 1)))
            else:
                dist = F.mse_loss(y_pred, y_true)
            return dist * self.rate

    return TemplateLoss


def SmoothLoss():
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate
            kernel = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
            kernel = torch.from_numpy(kernel).float()
            kernel = kernel.unsqueeze(0)
            kernel = torch.stack([kernel, kernel, kernel])
            self.kernel = nn.Parameter(kernel)
            self.kernel.requires_grad = False
            self.face_mask = nn.Parameter(face_mask.clone())
            self.face_mask.requires_grad = False

        def forward(self, y_pred):
            diff = F.conv2d(y_pred, self.kernel, padding=1, groups=3)
            dist = torch.sqrt(torch.sum(diff ** 2, 1))
            dist = dist * (self.face_mask * face_mask_mean_fix_rate)
            loss = torch.mean(dist)
            return loss * self.rate

    return TemplateLoss


def MaskLoss():
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate

        def forward(self, y_true, y_pred):
            return F.binary_cross_entropy(y_pred, y_true) * self.rate

    return TemplateLoss


def getLossFunction(loss_func_name='SquareError'):
    if loss_func_name == 'RootSquareError' or loss_func_name == 'rse':
        return UVLoss(is_foreface=False, is_weighted=False)
    elif loss_func_name == 'WeightedRootSquareError' or loss_func_name == 'wrse':
        return UVLoss(is_foreface=False, is_weighted=True)
    elif loss_func_name == 'ForefaceRootSquareError' or loss_func_name == 'frse':
        return UVLoss(is_foreface=True, is_weighted=False)
    elif loss_func_name == 'ForefaceWeightedRootSquareError' or loss_func_name == 'fwrse':
        return UVLoss(is_foreface=True, is_weighted=True)
    elif loss_func_name == 'nme':
        return UVLoss(is_foreface=True, is_weighted=False, is_nme=True)
    elif loss_func_name == 'mae':
        return ParamLoss('mae')
    elif loss_func_name == 'mse':
        return ParamLoss('mse')
    elif loss_func_name == 'rmse':
        return ParamLoss('rmse')
    elif loss_func_name == 'bce' or loss_func_name == 'BinaryCrossEntropy':
        return MaskLoss()
    else:
        print('unknown loss:', loss_func_name)


def PRNError(is_2d=False, is_normalized=True, is_foreface=True, is_landmark=False, is_gt_landmark=False, is_weighted_landmark=False):
    def templateError(y_true, y_pred, bbox=None, landmarks=None):
        assert (not (is_foreface and is_landmark))
        y_true = y_true.copy()
        y_pred = y_pred.copy()
        y_true[:, :, 2] = y_true[:, :, 2] * face_mask_np
        y_pred[:, :, 2] = y_pred[:, :, 2] * face_mask_np
        y_true_mean = np.mean(y_true[:, :, 2]) * face_mask_mean_fix_rate
        y_pred_mean = np.mean(y_pred[:, :, 2]) * face_mask_mean_fix_rate
        y_true[:, :, 2] = y_true[:, :, 2] - y_true_mean
        y_pred[:, :, 2] = y_pred[:, :, 2] - y_pred_mean

        if is_landmark:
            # the gt landmark is not the same as the landmarks get from mesh using index
            if is_gt_landmark:
                gt = landmarks.copy()
                gt[:, 2] = gt[:, 2] - y_true_mean

                if is_weighted_landmark:
                    pred = getWeightedKpt(y_pred)
                else:
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


def ICPError(is_2d=False, is_normalized=True, is_foreface=True):
    def templateError(y_true, y_pred, bbox=None, landmarks=None):
        y_true = y_true.copy()
        y_pred = y_pred.copy()

        y_true_vertices = []
        y_pred_vertices = []

        for i in range(256):
            for j in range(256):
                if face_mask_np[i, j] > 0:
                    y_true_vertices.append(y_true[i, j])
                    y_pred_vertices.append(y_pred[i, j])
        y_pred_vertices = np.array(y_pred_vertices)
        y_true_vertices = np.array(y_true_vertices)
        Tform, mean_dist, break_itr = icp(y_pred_vertices, y_true_vertices, max_iterations=30)

        # R = Tform[0:3, 0:3]
        # T = Tform[0:3, 3]
        #
        # y_pred = y_pred.dot(R.T)
        # y_pred = y_pred + T
        #
        # diff = np.square(y_true - y_pred)
        # if is_2d:
        #     dist = np.sqrt(np.sum(diff[:, :, 0:2], axis=-1))
        # else:
        #     # 3d
        #     dist = np.sqrt(np.sum(diff, axis=-1))
        # if is_foreface:
        #     dist = dist * face_mask_np * face_mask_mean_fix_rate

        if is_normalized:
            # bbox_size = np.sqrt(np.sum(np.square(bbox[0, :] - bbox[1, :])))
            # bbox_size = np.sqrt((bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1]))
            outer_interocular_dist = y_true[uv_kpt[36, 0], uv_kpt[36, 1]] - y_true[uv_kpt[45, 0], uv_kpt[45, 1]]
            bbox_size = np.sqrt(outer_interocular_dist[0:2].dot(outer_interocular_dist[0:2]))
        else:
            bbox_size = 1.
        loss = np.mean(mean_dist / bbox_size)
        return loss

    return templateError


def getErrorFunction(error_func_name='NME', rate=1.0):
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
    elif error_func_name == 'weightlandmark2d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True, is_weighted_landmark=True)
    elif error_func_name == 'weightlandmark3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True, is_weighted_landmark=True)
    elif error_func_name == 'icp':
        return ICPError(is_normalized=True)

    else:
        print('unknown error:', error_func_name)
