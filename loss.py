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
            foreface = y_pred * self.face_mask
            diff = F.conv2d(foreface, self.kernel, padding=1, groups=3)
            # dist = torch.sqrt(torch.sum(diff ** 2, 1))
            dist = torch.norm(diff, dim=1)
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


def KptLoss():
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate

        def forward(self, y_true, y_pred):
            dist = torch.mean(torch.sqrt(torch.sum((y_true[:, :, uv_kpt[:, 0], uv_kpt[:, 1]] - y_pred[:, :, uv_kpt[:, 0], uv_kpt[:, 1]]) ** 2, 1)))
            return dist * self.rate

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
    elif loss_func_name == 'smooth':
        return SmoothLoss()
    elif loss_func_name == 'kpt':
        return KptLoss()
    else:
        print('unknown loss:', loss_func_name)


def PRNError(is_2d=False, is_normalized=True, is_foreface=True, is_landmark=False, is_gt_landmark=False, is_weighted_landmark=False):
    def templateError(y_gt, y_fit, bbox=None, landmarks=None):
        assert (not (is_foreface and is_landmark))
        y_true = y_gt.copy()
        y_pred = y_fit.copy()
        y_true[:, :, 2] = y_true[:, :, 2] * face_mask_np
        y_pred[:, :, 2] = y_pred[:, :, 2] * face_mask_np
        y_true_mean = np.mean(y_true[:, :, 2]) * face_mask_mean_fix_rate
        y_pred_mean = np.mean(y_pred[:, :, 2]) * face_mask_mean_fix_rate
        y_true[:, :, 2] = y_true[:, :, 2] - y_true_mean
        y_pred[:, :, 2] = y_pred[:, :, 2] - y_pred_mean

        # y_true[:, :, 0] = y_true[:, :, 0] * face_mask_np
        # y_pred[:, :, 0] = y_pred[:, :, 0] * face_mask_np
        # y_true_mean0 = np.mean(y_true[:, :, 0]) * face_mask_mean_fix_rate
        # y_pred_mean0 = np.mean(y_pred[:, :, 0]) * face_mask_mean_fix_rate
        # y_true[:, :, 0] = y_true[:, :, 0] - y_true_mean0
        # y_pred[:, :, 0] = y_pred[:, :, 0] - y_pred_mean0
        #
        # y_true[:, :, 1] = y_true[:, :, 1] * face_mask_np
        # y_pred[:, :, 1] = y_pred[:, :, 1] * face_mask_np
        # y_true_mean1 = np.mean(y_true[:, :, 1]) * face_mask_mean_fix_rate
        # y_pred_mean1 = np.mean(y_pred[:, :, 1]) * face_mask_mean_fix_rate
        # y_true[:, :, 1] = y_true[:, :, 1] - y_true_mean1
        # y_pred[:, :, 1] = y_pred[:, :, 1] - y_pred_mean1

        if is_landmark:
            # the gt landmark is not the same as the landmarks get from mesh using index
            if is_gt_landmark:
                gt = landmarks.copy()
                gt[:, 2] = gt[:, 2] - gt[:, 2].mean()

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
                gt[:, 2] = gt[:, 2] - gt[:, 2].mean()
                pred[:, 2] = pred[:, 2] - pred[:, 2].mean()
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
            if is_landmark:
                bbox_size = np.sqrt((bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1]))
            else:
                face_vertices = y_gt[face_mask_np > 0]
                minx, maxx = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
                miny, maxy = np.min(face_vertices[:, 1]), np.max(face_vertices[:, 1])
                llength = np.sqrt((maxx - minx) * (maxy - miny))
                bbox_size = llength
        else:
            bbox_size = 1.
        loss = np.mean(dist / bbox_size)
        return loss

    return templateError


def cp(kpt_src, kpt_dst):
    A = kpt_src
    B = kpt_dst
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.T.dot(BB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # if np.linalg.det(R) < 0:
    #     print('singular R')
    #     Vt[2, :] *= -1
    #     R = Vt.T.dot(U.T)
    t = mu_B - mu_A.dot(R.T)
    tform = np.zeros((4, 4))
    tform[0:3, 0:3] = R
    tform[0:3, 3] = t
    tform[3, 3] = 1
    return tform


import numba


@numba.njit
def findNearestNB(src, dst):
    n = len(src)
    m = len(dst)
    out = np.zeros((n, 3))
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = np.linalg.norm(src[i])
    for i in range(n):
        for j in range(m):
            d = np.linalg.norm(dst[j] - src[i])
            if d < dist[i]:
                dist[i] = d
                out[i] = dst[j]
    return np.mean(dist)


from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=1)


def findNearest(src, dst, is_return_dist=False):
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    n = len(src)
    indices = indices.flatten()
    if is_return_dist:
        dist = np.zeros(n)
        for i in range(n):
            dist[i] = np.linalg.norm(src[i] - dst[indices[i]])
        return indices, dist
    else:
        return indices


def ICPError(is_interocular=True):
    def templateError(y_true, y_pred, bbox=None, landmarks=None):
        y_true = y_true.copy()
        y_pred = y_pred.copy()

        y_pred_vertices = y_pred[face_mask_np > 0]
        y_true_vertices = y_true[face_mask_np > 0]
        # Tform, mean_dist, break_itr = icp(y_pred_vertices[0::4], y_true_vertices[0::4], max_iterations=50)

        if is_interocular:
            Tform = cp(y_pred_vertices, y_true_vertices)
            # Tform, mean_dist, break_itr = icp(y_pred_vertices[0::], y_true_vertices[0::], max_iterations=5, init_pose=Tform)

            y_fit_vertices = y_pred_vertices.dot(Tform[0:3, 0:3].T) + Tform[0:3, 3]
            #
            dist = np.linalg.norm(y_fit_vertices - y_true_vertices, axis=1)
            # dist = mean_dist

            # ind = findNearest(y_true_vertices, y_fit_vertices)
            # dist=np.linalg.norm(y_true_vertices-y_fit_vertices[ind],axis=-1)
            # map_pred_vertices = y_pred_vertices[ind[:]]
            # dist = np.linalg.norm(map_pred_vertices- y_true_vertices,axis=-1)
            # dist=mean_dist

            outer_interocular_dist = y_true[uv_kpt[36, 0], uv_kpt[36, 1]] - y_true[uv_kpt[45, 0], uv_kpt[45, 1]]
            bbox_size = np.linalg.norm(outer_interocular_dist[0:3])

            loss = np.mean(dist / bbox_size)
        else:
            Tform = cp(y_pred_vertices, y_true_vertices)

            # # it is 2D interocular dist in PRN while 2DASl using 2D bboxsize
            # fit_pred=y_pred_vertices
            # fit_pred[:,2]=fit_pred[:,2]-fit_pred[:,2].mean()+y_true_vertices[:,2].mean()
            y_fit_vertices = y_pred_vertices.dot(Tform[0:3, 0:3].T) + Tform[0:3, 3]
            dist = np.sqrt(np.sum((y_fit_vertices - y_true_vertices) ** 2, axis=-1))
            minx, maxx = np.min(y_true[:, :, 0]), np.max(y_true[:, :, 0])
            miny, maxy = np.min(y_true[:, :, 1]), np.max(y_true[:, :, 1])
            llength = np.sqrt((maxx - minx) * (maxy - miny))
            bbox_size = llength
            loss = np.mean(dist / bbox_size)
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
        return ICPError(is_interocular=True)
    elif error_func_name == 'icp2':
        return ICPError(is_interocular=False)

    else:
        print('unknown error:', error_func_name)
