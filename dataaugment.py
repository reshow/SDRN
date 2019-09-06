import numpy as np
from skimage import io, transform
import math
import copy
from PIL import ImageEnhance, ImageOps, ImageFile, Image


def randomColor(image):
    """
    """
    PIL_image = Image.fromarray((image * 255.).astype(np.uint8))
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(PIL_image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.
    out = np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))
    out = out / 255.
    return out


def rotateData(x, y):
    angle = np.random.randint(-90, 90)
    angle = angle / 180. * np.pi
    [image_height, image_width, image_channel] = x.shape
    # move-rotate-move
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0], [math.sin(-angle), math.cos(angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rform = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0], [math.sin(angle), math.cos(-angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rform_inv = t2.dot(r1).dot(t1)

    rotate_x = transform.warp(x, rform_inv,
                              output_shape=(image_height, image_width))
    rotate_y = y.copy()
    rotate_y[:, :, 2] = 1.
    rotate_y = rotate_y.reshape(image_width * image_height, image_channel)
    rotate_y = rotate_y.dot(rform.T)
    rotate_y = rotate_y.reshape(image_height, image_width, image_channel)
    rotate_y[:, :, 2] = y[:, :, 2]
    # for i in range(image_height):
    #     for j in range(image_width):
    #         rotate_y[i][j][2] = 1.
    #         rotate_y[i][j] = rotate_y[i][j].dot(rform.T)
    #         rotate_y[i][j][2] = y[i][j][2]
    return rotate_x, rotate_y


def gaussNoise(x, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, x.shape)
    out = x + noise
    out = np.clip(out, 0., 1.0)
    # cv.imshow("gasuss", out)
    return out
