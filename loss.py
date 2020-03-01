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
weight_mask_np[weight_mask_np == 4] = 12

weight_mask = torch.from_numpy(weight_mask_np)
face_mask = torch.from_numpy(face_mask_np)
face_mask_3D = toTensor(np.repeat(np.reshape(face_mask_np, (256, 256, 1)), 3, -1))
foreface_ind = np.array(np.where(face_mask_np > 0)).T
if torch.cuda.is_available():
    weight_mask = weight_mask.cuda().float()
    face_mask = face_mask.cuda().float()
    face_mask_3D = face_mask_3D.cuda().float()
micc_face_mask = io.imread('uv-data/uv_face_MICC.png').astype(float)
micc_face_mask[micc_face_mask > 0] = 1
micc_center_mask = io.imread('uv-data/uv_face_MICC_coreface.png')
micc_center_mask[micc_center_mask > 0] = 1


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
                pred = y_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
                gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
                for i in range(y_true.shape[0]):
                    pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
                    gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
                dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
                left = torch.min(gt[:, 0, :], dim=1)[0]
                right = torch.max(gt[:, 0, :], dim=1)[0]
                top = torch.min(gt[:, 1, :], dim=1)[0]
                bottom = torch.max(gt[:, 1, :], dim=1)[0]
                bbox_size = torch.sqrt((right - left) * (bottom - top))
                dist = dist / bbox_size
                return torch.mean(dist) * self.rate

            dist = torch.sqrt(torch.sum((y_true - y_pred) ** 2, 1))
            if is_weighted:
                dist = dist * self.weight_mask
            if is_foreface:
                dist = dist * (self.face_mask * face_mask_mean_fix_rate)

            loss = torch.mean(dist)
            return loss * self.rate

    return TemplateLoss


def UVLoss2(is_foreface=False, is_weighted=False, is_kpt=False):
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate

            temp_weight_mask = weight_mask.clone()
            if is_kpt:
                temp_weight_mask = (temp_weight_mask ** 2) / 4.
            self.weight_mask = nn.Parameter(temp_weight_mask.clone())

            self.face_mask = nn.Parameter(face_mask.clone())
            self.weight_mask.requires_grad = False
            self.face_mask.requires_grad = False

        def forward(self, y_true, y_pred):
            dist = torch.sum((y_true - y_pred) ** 2, 1)
            if is_weighted:
                dist = dist * self.weight_mask
            if is_foreface:
                dist = dist * (self.face_mask * face_mask_mean_fix_rate)

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
            elif mode == 'maemod':
                dist = y_true - y_pred
                for i in range(y_true.shape[0]):
                    for j in range(y_true.shape[1]):
                        if dist[i][j] > 1:
                            dist[i][j] = dist[i][j] - 2
                        if dist[i][j] < -1:
                            dist[i][j] = dist[i][j] + 2
                dist = torch.mean(torch.abs(dist))
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


def SecondOrderLoss():
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

        def forward(self, y_true, y_pred):
            foreface_pred = y_pred * self.face_mask
            diff_pred = F.conv2d(foreface_pred, self.kernel, padding=1, groups=3)
            foreface_true = y_true * self.face_mask
            diff_true = F.conv2d(foreface_true, self.kernel, padding=1, groups=3)
            diff = diff_pred - diff_true
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


def KptLoss(is_centralize=False):
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate

        def forward(self, y_true, y_pred):
            if is_centralize:
                gt = y_true[:, :, uv_kpt[:, 0], uv_kpt[:, 1]]
                pred = y_pred[:, :, uv_kpt[:, 0], uv_kpt[:, 1]]
                for i in range(y_true.shape[0]):
                    pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
                    gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
                dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
                left = torch.min(gt[:, 0, :], dim=1)[0]
                right = torch.max(gt[:, 0, :], dim=1)[0]
                top = torch.min(gt[:, 1, :], dim=1)[0]
                bottom = torch.max(gt[:, 1, :], dim=1)[0]
                bbox_size = torch.sqrt((right - left) * (bottom - top))
                dist = dist / bbox_size
            else:
                dist = torch.mean(torch.sqrt(torch.sum((y_true[:, :, uv_kpt[:, 0], uv_kpt[:, 1]] - y_pred[:, :, uv_kpt[:, 0], uv_kpt[:, 1]]) ** 2, 1)))
            return dist * self.rate

    return TemplateLoss


def AttentedKptLoss():
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate

        def forward(self, y_true, y_pred, attention):
            gt = y_true[:, :, uv_kpt[:, 0], uv_kpt[:, 1]]
            pred = y_pred[:, :, uv_kpt[:, 0], uv_kpt[:, 1]]
            dist = torch.norm(pred - gt, dim=1)
            dist1 = dist.clone()
            attention2 = attention.detach()
            for i in range(y_true.shape[0]):
                shallow_kpt_args = torch.argsort(gt[i, 2, :])[34:]
                dist1[i, shallow_kpt_args] = dist[i, shallow_kpt_args] * 2
                W = torch.ones(68, device=y_true.device)
                for j in range(68):
                    t1 = min(max(int(gt[i, 1, j] / 8 * 280), 0), 31)
                    t2 = min(max(int(gt[i, 0, j] / 8 * 280), 0), 31)
                    W[j] = W[j] * attention2[i, 0, t1, t2]
                    # dist1[i, j] = dist[i, j] * attention[i, 0, t1, t2]
                dist1[i] = dist[i] * W
            dist2 = torch.mean(dist)
            return dist2 * self.rate

    return TemplateLoss


def AlignmentLoss(is_centralize=False):
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate

        def forward(self, y_true, y_pred):
            if is_centralize:
                gt = y_true.clone()
                pred = y_pred.clone()
                for i in range(y_true.shape[0]):
                    pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
                    gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
                dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
                left = torch.min(gt[:, 0, :], dim=1)[0]
                right = torch.max(gt[:, 0, :], dim=1)[0]
                top = torch.min(gt[:, 1, :], dim=1)[0]
                bottom = torch.max(gt[:, 1, :], dim=1)[0]
                bbox_size = torch.sqrt((right - left) * (bottom - top))
                dist = dist / bbox_size
            else:
                dist = torch.mean(torch.sqrt(torch.sum((y_true - y_pred) ** 2, 1)))
            return dist * self.rate

    return TemplateLoss


def AttendedAlignmentLoss():
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate

        def forward(self, y_true, y_pred, attention):
            gt = y_true
            pred = y_pred
            dist = torch.norm(pred - gt, dim=1)
            dist1 = dist.clone()
            attention2 = attention.detach()
            for i in range(y_true.shape[0]):
                shallow_kpt_args = torch.argsort(gt[i, 2, :])[34:]
                dist1[i, shallow_kpt_args] = dist[i, shallow_kpt_args] * 2
                W = torch.ones(68, device=y_true.device)
                for j in range(68):
                    t1 = min(max(int(gt[i, 1, j] / 8 * 280), 0), 31)
                    t2 = min(max(int(gt[i, 0, j] / 8 * 280), 0), 31)
                    W[j] = W[j] * attention2[i, 0, t1, t2]
                    # dist1[i, j] = dist[i, j] * attention[i, 0, t1, t2]
                dist1[i] = dist[i] * W
            dist2 = torch.mean(dist)
            return dist2 * self.rate

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
    elif loss_func_name == 'fwse':
        return UVLoss2(is_foreface=True, is_weighted=True, is_kpt=False)
    elif loss_func_name == 'fwsekpt':
        return UVLoss2(is_foreface=True, is_weighted=True, is_kpt=True)
    elif loss_func_name == 'nme':
        return UVLoss(is_foreface=True, is_weighted=False, is_nme=True)
    elif loss_func_name == 'mae':
        return ParamLoss('mae')
    elif loss_func_name == 'maemod':
        return ParamLoss('maemod')
    elif loss_func_name == 'mse':
        return ParamLoss('mse')
    elif loss_func_name == 'rmse':
        return ParamLoss('rmse')
    elif loss_func_name == 'bce' or loss_func_name == 'BinaryCrossEntropy':
        return MaskLoss()
    elif loss_func_name == 'smooth':
        return SmoothLoss()
    elif loss_func_name == '2nd':
        return SecondOrderLoss()
    elif loss_func_name == 'kpt':
        return KptLoss()
    elif loss_func_name == 'kptc':
        return KptLoss(is_centralize=True)
    elif loss_func_name == 'akpt':
        return AttentedKptLoss()
    elif loss_func_name == 'align':
        return AlignmentLoss(is_centralize=False)
    elif loss_func_name == 'alignc':
        return AlignmentLoss(is_centralize=True)
    elif loss_func_name == 'aalign':
        return AttendedAlignmentLoss()
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

        if is_normalized:  # 2D bbox size
            # bbox_size = np.sqrt(np.sum(np.square(bbox[0, :] - bbox[1, :])))
            if is_landmark:
                bbox_size = np.sqrt((bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1]))
            else:
                face_vertices = y_gt[face_mask_np > 0]
                minx, maxx = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
                miny, maxy = np.min(face_vertices[:, 1]), np.max(face_vertices[:, 1])
                llength = np.sqrt((maxx - minx) * (maxy - miny))
                bbox_size = llength
        else:  # 3D bbox size
            face_vertices = y_gt[face_mask_np > 0]
            minx, maxx = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
            miny, maxy = np.min(face_vertices[:, 1]), np.max(face_vertices[:, 1])
            minz, maxz = np.min(face_vertices[:, 2]), np.max(face_vertices[:, 2])
            if is_landmark:
                llength = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
            else:
                llength = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2 + (maxz - minz) ** 2)
            bbox_size = llength

        loss = np.mean(dist / bbox_size)
        return loss

    return templateError


def cp(kpt_src, kpt_dst, is_scale=True):
    if is_scale:
        sum_dist1 = np.sum(np.linalg.norm(kpt_src - kpt_src[0], axis=1))
        sum_dist2 = np.sum(np.linalg.norm(kpt_dst - kpt_dst[0], axis=1))
        A = kpt_src * sum_dist2 / sum_dist1
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
        R = R * sum_dist2 / sum_dist1
    else:
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


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def myICP(C, B, init_pose=None, max_iterations=20, tolerance=0.1, is_valid=False):
    if is_valid:
        A = []
        distances, indices = nearest_neighbor(C, B)
        for i in range(len(distances)):
            if distances[i] < 10:
                A.append(C[i])
        A = np.array(A)
    else:
        A = C

    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    # init

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T = cp(src[:m, :].T, dst[:m, indices].T, is_scale=True)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T = cp(A, src[:m, :].T, is_scale=True)

    return T, distances, i


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


@numba.njit
def renderBG(true_bg, pred_bg, true_vert, pred_vert):
    for i in range(len(true_vert)):
        x = int(true_vert[i][0])
        y = int(true_vert[i][1])
        z = int(true_vert[i][2])
        if 1 < x < 254 and 1 < y < 254:
            true_bg[x, y] = max(true_bg[x, y], z)
            true_bg[x - 1, y] = max(true_bg[x - 1, y], z)
            true_bg[x - 1, y - 1] = max(true_bg[x - 1, y - 1], z)
            true_bg[x - 1, y + 1] = max(true_bg[x - 1, y + 1], z)
            true_bg[x, y - 1] = max(true_bg[x, y - 1], z)
            true_bg[x, y + 1] = max(true_bg[x, y + 1], z)
            true_bg[x + 1, y - 1] = max(true_bg[x + 1, y - 1], z)
            true_bg[x + 1, y] = max(true_bg[x + 1, y], z)
            true_bg[x + 1, y + 1] = max(true_bg[x + 1, y + 1], z)
    for i in range(len(pred_vert)):
        x = int(pred_vert[i][0])
        y = int(pred_vert[i][1])
        z = int(pred_vert[i][2])
        if 1 < x < 254 and 1 < y < 254:
            pred_bg[x, y] = max(pred_bg[x, y], z)
            pred_bg[x - 1, y] = max(pred_bg[x - 1, y], z)
            pred_bg[x - 1, y - 1] = max(pred_bg[x - 1, y - 1], z)
            pred_bg[x - 1, y + 1] = max(pred_bg[x - 1, y + 1], z)
            pred_bg[x, y - 1] = max(pred_bg[x, y - 1], z)
            pred_bg[x, y + 1] = max(pred_bg[x, y + 1], z)
            pred_bg[x + 1, y - 1] = max(pred_bg[x + 1, y - 1], z)
            pred_bg[x + 1, y] = max(pred_bg[x + 1, y], z)
            pred_bg[x + 1, y + 1] = max(pred_bg[x + 1, y + 1], z)


def depthAlign(y_true, y_pred):
    kpt = y_pred[uv_kpt[:, 0], uv_kpt[:, 1]]
    valid_pred = y_pred[face_mask_np > 0]

    true_bg = np.zeros((256, 256))
    pred_bg = np.zeros((256, 256))

    renderBG(true_bg, pred_bg, y_true, valid_pred)

    sum_true = 0
    sum_pred = 0
    for p in kpt:
        x = int(p[0])
        y = int(p[1])
        sum_true += true_bg[x, y]
        sum_pred += pred_bg[x, y]
    print(sum_true, sum_pred)
    return sum_true / 68., sum_pred / 68.


def MICCError():
    def templateError(y_true, y_pred):
        mean_depth_true, mean_depth_pred = depthAlign(y_true, y_pred)

        y_true_vertices = y_true.copy()
        y_pred_vertices = y_pred[micc_center_mask > 0]

        y_true_vertices[:, 2] = y_true_vertices[:, 2] - mean_depth_true
        y_pred_vertices[:, 2] = y_pred_vertices[:, 2] - mean_depth_pred

        y_pred_vertices = y_pred_vertices[::4]

        # Tform = cp(y_pred_vertices, y_true_vertices)

        Tform, mean_dist, break_itr = myICP(y_pred_vertices[0::], y_true_vertices[0::], max_iterations=40, tolerance=0.1)
        y_fit_vertices = y_pred_vertices.dot(Tform[0:3, 0:3].T) + Tform[0:3, 3]
        # y_fit_vertices = y_pred_vertices + Tform[0:3, 3]
        # distances, ind = nearest_neighbor(y_fit_vertices, y_true_vertices)
        # dist = np.linalg.norm(y_true_vertices[ind] - y_fit_vertices, axis=-1)
        distances, ind = nearest_neighbor(y_true_vertices, y_fit_vertices)

        valid_true_vertices = []
        for i in range(len(distances)):
            if distances[i] < 5:
                valid_true_vertices.append(y_true_vertices[i])
        print('len', len(valid_true_vertices))
        valid_true_vertices = np.array(valid_true_vertices)[::4]
        valid_pred_vertices = y_pred[face_mask_np > 0].dot(Tform[0:3, 0:3].T) + Tform[0:3, 3]
        Tform, mean_dist, break_itr = myICP(valid_true_vertices, valid_pred_vertices, max_iterations=60, tolerance=0.1)

        # dist = np.linalg.norm(y_fit_vertices - y_true_vertices, axis=1)
        outer_interocular_dist = y_pred[uv_kpt[36, 0], uv_kpt[36, 1]] - y_pred[uv_kpt[45, 0], uv_kpt[45, 1]]
        bbox_size = np.linalg.norm(outer_interocular_dist[0:3])

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
        return ICPError(is_interocular=True)
    elif error_func_name == 'icp2':
        return ICPError(is_interocular=False)
    elif error_func_name == 'micc':
        return MICCError()
    elif error_func_name == 'mmfacekpt':
        return PRNError(is_2d=True, is_normalized=False, is_foreface=False, is_landmark=True)
    elif error_func_name == 'mmface3d':
        return PRNError(is_2d=False, is_normalized=False, is_foreface=True, is_landmark=False)

    else:
        print('unknown error:', error_func_name)
