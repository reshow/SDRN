import time
import numpy as np
import scipy.io as sio
from skimage import io, transform
import math
from dataloader import ImageData
import os
import random
import multiprocessing
from augmentation import prnAugment_keras
import threading
from masks import getImageAttentionMask, getVisibilityMask, face_mask_np3d

train_data = []


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class FitGenerator:
    def __init__(self, all_image_data):
        self.all_image_data = all_image_data
        self.next_index = 0
        self.image_height = 256
        self.image_width = 256
        self.image_channel = 3
        self.read_img = False
        self.read_posmap = False
        self.augment = False

    def get(self, batch_size=64, gen_mode='random'):
        x = []
        y = []
        if gen_mode == 'random':
            batch_num = batch_size
            indexs = np.random.randint(len(self.all_image_data), size=batch_size)
        elif gen_mode == 'order':
            if self.next_index == 0:
                # print('random shuffle')
                random.shuffle(self.all_image_data)
            if batch_size > len(self.all_image_data):
                batch_size = len(self.all_image_data)
            batch_num = batch_size
            if self.next_index + batch_size >= len(self.all_image_data):
                batch_num = len(self.all_image_data) - self.next_index
            indexs = np.array(range(self.next_index, self.next_index + batch_num))
            # print(self.next_index,self.next_index+batch_num)
            self.next_index = (self.next_index + batch_num) % len(self.all_image_data)
        else:
            indexs = None
            batch_num = 0
            print('unknown generate mode')
        for i in range(batch_num):
            image_path = self.all_image_data[indexs[i]].cropped_image_path
            image = io.imread(image_path) / 255.
            # image = transform.resize(image, (self.image_height, self.image_width, self.image_channel))
            # image = unchangeAugment(image)
            pos_path = self.all_image_data[indexs[i]].cropped_posmap_path
            pos = np.load(pos_path)
            pos = pos / 256.
            x.append(image)
            y.append(pos)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def worker(self, indexes):
        # print(indexes)
        x = []
        y = []
        for index in indexes:
            # print(index)
            if self.read_img:
                image_path = self.all_image_data[index].cropped_image_path
                # image_path='data/images/AFLW2000-crop-offset/image00002/image00002_cropped.jpg'
                image = io.imread(image_path) / 255.
                # image = transform.resize(image, (self.image_height, self.image_width, self.image_channel))
                # image = unchangeAugment(image)
                if self.augment:
                    image = prnAugment_keras(image)
                x.append(image)
            if self.read_posmap:
                pos_path = self.all_image_data[index].cropped_posmap_path
                # pos_path='data/images/AFLW2000-crop-offset/image00002/image00002_cropped_uv_posmap.npy'
                pos = np.load(pos_path)
                pos = pos / 256.
                y.append(pos)
                # if np.max(pos[:, :, 2]) > 1:
                #     print(np.max(pos[:, :, 2]))
        print('finish')
        return x, y

    def multiget(self, batch_size=64, gen_mode='random', worker_num=4):
        x = []
        y = []
        if gen_mode == 'random':
            batch_num = batch_size
            indexes = np.random.randint(len(self.all_image_data), size=batch_size)
        elif gen_mode == 'order':
            if self.next_index == 0:
                # print('random shuffle')
                random.shuffle(self.all_image_data)
            if batch_size > len(self.all_image_data):
                batch_size = len(self.all_image_data)
            batch_num = batch_size
            if self.next_index + batch_size >= len(self.all_image_data):
                batch_num = len(self.all_image_data) - self.next_index
            indexes = np.array(range(self.next_index, self.next_index + batch_num))
            # print(self.next_index,self.next_index+batch_num)
            self.next_index = (self.next_index + batch_num) % len(self.all_image_data)
        else:
            indexes = None
            batch_num = 0
            print('unknown generate mode')
        task_per_worker = math.ceil(batch_num / worker_num)
        st_idx = [task_per_worker * i for i in range(worker_num)]
        ed_idx = [min(batch_num, task_per_worker * (i + 1)) for i in range(worker_num)]
        # q = multiprocessing.Queue()
        # jobs = []
        # for i in range(worker_num):
        #     print(st_idx[i], ed_idx[i])
        #     idx = indexes[st_idx[i]:ed_idx[i]]
        #     p = multiprocessing.Process(target=self.worker, args=(idx, q))
        #     jobs.append(p)
        #     p.start()
        # for p in jobs:
        #     p.join()
        # for p in jobs:
        #     [xx, yy] = q.get()
        #     x.extend(xx)
        #     y.extend(yy)
        jobs = []
        for i in range(worker_num):
            # print(st_idx[i], ed_idx[i])
            idx = np.array(indexes[st_idx[i]:ed_idx[i]])
            p = MyThread(func=self.worker, args=(idx,))
            jobs.append(p)
        for p in jobs:
            p.start()
        for p in jobs:
            p.join()
        for p in jobs:
            [xx, yy] = p.get_result()
            x.extend(xx)
            y.extend(yy)
        x = np.array(x)
        y = np.array(y)
        return x, y


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


def changeImageType(data_dir):
    i = 0
    j=0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_name = file.split('.')[0]
            file_type = file.split('.')[1]
            if file_type == 'jpg' and 'cropped' in file_name:
                image = np.load(root + '/' + str(file_name) + '.npy')
                posmap = np.load(root + '/' + str(file_name) + '_uv_posmap.npy')

                posmap = posmap * face_mask_np3d
                if posmap.min() < 0 or posmap.max()>256.:
                    j+=1
                    print(posmap.min(),posmap.max(),j)
                # mask = getImageAttentionMask(image, posmap)
                # np.save(root + '/' + str(file_name).replace('_cropped', '_attention_mask.npy'), mask)

                # visibility_mask = getVisibilityMask(posmap, image.shape)
                # np.save(root + '/' + str(file_name).replace('_cropped', '_visibility_mask.npy'), visibility_mask.astype(np.uint8))

                # np.save(root + '/' + str(file_name) + '.npy',image.astype(np.uint8))

                print(i, file_name, end='\r')
                i += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='data preprocess arguments')

    parser.add_argument('-i', '--inputDir', default='data/images/AFLW2000-full', type=str,
                        help='path to the input directory, where input images are stored.')
    conf = parser.parse_args()
    changeImageType(conf.inputDir)
# addImageData('data/images/AFLW2000-crop')
# fg = FitGenerator(train_data)
# fg.read_img = True
# fg.augment = True
# fg.read_posmap=True
# t1 = time.clock()
# print(t1)
# for _ in range(30):
#     fg.get(64, 'order')
# t2 = time.clock()
# t = t2 - t1
# print(t)
