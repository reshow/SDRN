import time
import numpy as np
import scipy.io as sio
from skimage import io, transform
import math
from data import ImageData
import os
import random
import multiprocessing

train_data = []


class FitGenerator:
    def __init__(self, all_image_data):
        self.all_image_data = all_image_data
        self.next_index = 0
        self.image_height = 256
        self.image_width = 256
        self.image_channel = 3

    def get(self, batch_size=64, gen_mode='random'):
        """

        :param batch_size:
        :param gen_mode: random or order
        :return:
        """

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
            pos_path = self.all_image_data[indexs[i]].cropped_posmap_path
            pos = np.load(pos_path)
            pos = pos / 256.
            x.append(image)
            y.append(pos)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def worker(self, indexes, q):
        print(indexes)
        x = []
        y = []
        for index in indexes:
            print(index)
            image_path = self.all_image_data[index].cropped_image_path
            image = io.imread(image_path) / 255.
            # image = transform.resize(image, (self.image_height, self.image_width, self.image_channel))
            pos_path = self.all_image_data[index].cropped_posmap_path
            pos = np.load(pos_path)
            pos = pos / 256.
            x.append(image)
            y.append(pos)
        print('finish')
        q.put([x, y])

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

        q = multiprocessing.Queue()
        jobs = []
        for i in range(worker_num):
            print(st_idx[i], ed_idx[i])
            idx = indexes[st_idx[i]:ed_idx[i]]
            p = multiprocessing.Process(target=self.worker, args=(idx, q))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        for p in jobs:
            [xx, yy] = q.get()
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


if __name__ == '__main__':
    addImageData('data/images/AFLW2000-crop')
    fg = FitGenerator(train_data)
    t1 = time.time()
    for _ in range(10):
        fg.multiget(64, 'order')
    t2 = time.time()
    t = t2 - t1
    print(t)
