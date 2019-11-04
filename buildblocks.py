from dataloader import ImageData
import numpy as np
import pickle
import argparse
import os
import math
import threading
import scipy.io as sio
import ast

all_data = []
NUM_BLOCKS = 630
if not os.path.exists('data/images/td/'):
    os.mkdir('data/images/td/')
data_block_names = ['data/images/td/block' + str(i) + '.pkl' for i in range(NUM_BLOCKS)]


def addImageData(data_dir):
    temp_all_data = []
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            image_name = dir_name
            if not os.path.exists(root + '/' + dir_name + '/' + image_name + '_cropped.jpg'):
                print('skip ', root + '/' + dir_name)
                continue
            temp_image_data = ImageData()
            temp_image_data.readPath(root + '/' + dir_name)
            temp_all_data.append(temp_image_data)
    print(len(temp_all_data), 'data added')
    all_data.extend(temp_all_data)


def saveBlock(data_list, worker_id):
    i = 0
    for temp_data in data_list:
        i += 1
        print('worker', worker_id, 'task', i, end='\r')
        temp_data.image = np.load(temp_data.cropped_image_path).astype(np.uint8)
        temp_data.posmap = np.load(temp_data.cropped_posmap_path).astype(np.float16)
        temp_data.offset_posmap = np.load(temp_data.offset_posmap_path).astype(np.float16)
        temp_data.bbox_info = sio.loadmat(temp_data.bbox_info_path)
        temp_data.attention_mask = np.load(temp_data.attention_mask_path).astype(np.uint8)

    print('saving data block', worker_id)
    if os.path.exists('data/images/td/block' + str(worker_id) + '.pkl'):
        print('data path list saved', worker_id)
        return
    f = open('data/images/td/block' + str(worker_id) + '.pkl', 'wb')
    pickle.dump(data_list, f)
    f.close()
    print('data path list saved', worker_id)


def multiSaveBlock(st, ed):
    worker_num = NUM_BLOCKS

    total_task = len(all_data)
    import random
    random.seed(0)
    random.shuffle(all_data)
    jobs = []
    task_per_worker = math.ceil(total_task / worker_num)
    st_idx = [task_per_worker * i for i in range(worker_num)]
    ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(worker_num)]
    for i in range(st, ed):
        # temp_data_processor = copy.deepcopy(data_processor)
        p = threading.Thread(target=saveBlock, args=(
            all_data[st_idx[i]:ed_idx[i]], i))
        jobs.append(p)
        p.start()
        print('start ', i)
        if (i - st) % 4 == 3:
            for p in jobs:
                p.join()
        print('batch end')

    print('all start')


def addMoreBlock(st, worker_num=30):
    import random
    random.seed(0)
    total_task = len(all_data)
    random.shuffle(all_data)
    jobs = []
    task_per_worker = math.ceil(total_task / worker_num)
    st_idx = [task_per_worker * i for i in range(worker_num)]
    ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(worker_num)]
    for i in range(worker_num):
        # temp_data_processor = copy.deepcopy(data_processor)
        p = threading.Thread(target=saveBlock, args=(
            all_data[st_idx[i]:ed_idx[i]], st + i))
        jobs.append(p)
        p.start()
        print('start ', i)
        if (i - st) % 4 == 3:
            for p in jobs:
                p.join()
        print('batch end')

    print('all start')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model arguments')
    parser.add_argument('-td', '--trainDataDir', nargs='+', type=str, help='training image directories')
    parser.add_argument('-s', '--start', default=300, type=int,
                        help='')
    parser.add_argument('-e', '--end', default=300, type=int,
                        help='')
    parser.add_argument('--more', default=False, type=ast.literal_eval)

    run_args = parser.parse_args()

    if not run_args.more:
        ft = open('data' + '/' + 'train_data.pkl', 'rb')
        data1 = pickle.load(ft)
        all_data.extend(data1)
        # for dir_x in run_args.trainDataDir:
        #     addImageData(dir_x)

        multiSaveBlock(run_args.start, run_args.end)
    else:
        for dir_x in run_args.trainDataDir:
            addImageData(dir_x)
        addMoreBlock(run_args.start, run_args.end - run_args.start)
