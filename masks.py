import numpy as np
import numba
from data import UVmap2Mesh, uv_kpt, bfm2Mesh, getLandmark, mesh2UVmap, bfm, face_mask_np, matrix2Angle
from faceutil import mesh
import cv2
import numba
from PIL import Image

face_mask_np3d = np.stack([face_mask_np, face_mask_np, face_mask_np], axis=2)


# for init image visibility
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


@numba.njit
def isPointInTriangle(p, p0, p1, p2):
    # from cython core
    v0 = p2 - p0
    v1 = p1 - p0
    v2 = p - p0
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)
    if dot00 * dot11 - dot01 * dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno
    return (u >= 0) and (v >= 0) and (u + v < 1)


# for uvmap visibility
@numba.njit
def getTriangbleBuffer(all_triangles, posmap_around, height, width, depth_buffer, triangle_buffer):
    for id in range(len(all_triangles)):
        t = all_triangles[id]
        p0 = posmap_around[t[0, 0], t[0, 1]]
        p1 = posmap_around[t[1, 0], t[1, 1]]
        p2 = posmap_around[t[2, 0], t[2, 1]]
        y_min = int(min(p0[0], p1[0], p2[0]))
        y_max = int(max(p0[0], p1[0], p2[0]))
        x_min = int(min(p0[1], p1[1], p2[1]))
        x_max = int(max(p0[1], p1[1], p2[1]))
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                p = np.array([y, x])
                if p[0] < 2 or p[0] > height - 3 or p[1] < 2 or p[1] > width - 3 or isPointInTriangle(p, p0[:2], p1[:2], p2[:2]):
                    p_depth = p0[2] / 3 + p1[2] / 3 + p2[2] / 3
                    if p_depth > depth_buffer[x, y]:
                        depth_buffer[x, y] = p_depth
                        triangle_buffer[x, y] = id


def getVisibilityMask(posmap, image_shape,downsample_stride = 4):

    posmap_around = (np.around(posmap / downsample_stride * face_mask_np3d).clip(1, 254)).astype(np.uint8)
    posmap_around = Image.fromarray(posmap_around)
    posmap_around = posmap_around.resize((64, 64), Image.NEAREST)
    posmap_around = np.array(posmap_around).astype(np.float32)

    [height, width, channel] = image_shape
    height = int(height / downsample_stride)
    width = int(width / downsample_stride)
    [uv_h, uv_w, uv_c] = posmap_around.shape

    depth_buffer = np.zeros((height, width))
    depth_buffer = depth_buffer - 100000
    triangle_buffer = np.zeros((height, width))
    triangle_buffer = triangle_buffer - 1
    visibility = np.zeros((height, width))

    all_triangles = []
    for i in range(uv_h):
        for j in range(uv_w):
            if (i > 0) & (i < height - 1) & (j < width - 1):
                if face_mask_np[i, j] == 0 or face_mask_np[i, j + 1] == 0 or face_mask_np[i - 1, j] == 0 or face_mask_np[i + 1, j + 1] == 0:
                    continue
                triangles = np.array([[[i, j], [i, j + 1], [i - 1, j]], [[i, j], [i, j + 1], [i + 1, j + 1]]])
                for t in triangles:
                    all_triangles.append(t)
    all_triangles = np.array(all_triangles)
    getTriangbleBuffer(all_triangles, posmap_around, height, width, depth_buffer, triangle_buffer)

    for i in range(height):
        for j in range(width):
            if triangle_buffer[i, j] < 0:
                continue
            t = all_triangles[int(triangle_buffer[i, j])]
            visibility[t[0, 0], t[0, 1]] = 1
            visibility[t[1, 0], t[1, 1]] = 1
            visibility[t[2, 0], t[2, 1]] = 1

    visibility = Image.fromarray(visibility.astype(np.uint8))
    [height, width, channel] = image_shape
    visibility = visibility.resize((height, width), Image.NEAREST)
    visibility = np.array(visibility).astype(np.uint8)
    return visibility


def getAngleVisibility(R, posmap_shape):
    [height, width, channel] = posmap_shape
    x, y, z = matrix2Angle(R)
    visibility = np.zeros((height, width)).astype(np.uint8)
    left = 0
    right = width

    yaw_rate = y / np.pi

    while yaw_rate < -1:
        yaw_rate += 2
    while yaw_rate > 1:
        yaw_rate -= 2

    left = int(left + yaw_rate * width)
    right = int(right + yaw_rate * width)

    left = max(left, 0)
    left = min(left, width - 1)
    right = max(right, 1)
    right = min(right, width)
    visibility[:, left:right] = 1
    visibility = visibility * face_mask_np
    return visibility.astype(np.uint8)


if __name__ == '__main__':
    from dataloader import ImageData
    import time, visualize

    a = ImageData()
    a.readPath('data/images/AFLW2000-crop/image00004')
    a.readFile('offset')
    trans_mat = a.bbox_info['TformOffset']
    R = trans_mat[0:3, 0:3]
    mask = getAngleVisibility(R, posmap_shape=a.posmap.shape)
    visualize.showImage(mask)

    # print(time.time())
    # for i in range(30):
    #     mask = getVisibilityMask(posmap, image.shape)
    #     # visualize.showImage(mask)
    #     print(i)
    # print(time.time())
