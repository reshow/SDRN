import time
import numpy as np
import scipy.io as sio
from skimage import io, transform
from data import FitGenerator
import argparse
from run import NetworkManager
import math
from data import ImageData
import os

train_data = []


def addImageData(data_dir):
    all_data = []
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            image_name = dir_name
            if not os.path.exists(root + '/' + dir_name + '/' + image_name + '_cropped.jpg'):
                print('skip ', root + '/' + dir_name)
                continue
            temp_image_data = ImageData()
            temp_image_data.readPath(root + '/' + dir_name)
            all_data.append(temp_image_data)

    train_data.extend(all_data)


if __name__ == '__main__':
    addImageData('data/images/AFLW2000-crop')
    fg = FitGenerator(train_data)
    t1 = time.time()
    for i in range(10):
        fg.get(64, 'order')
    t2 = time.time()
    t = t2 - t1
