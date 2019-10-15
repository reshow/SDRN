import os
import math
import argparse
from dataloader import ImageData
import multiprocessing

train_data = []


def addFlushData(data_dir):
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
    print(len(all_data), 'data added')
    train_data.extend(all_data)


def loadImageData(all_data):
    i = 0
    for d in all_data:
        print(i, d.getImage().shape, d.getPosmap().shape, d.getAttentionMask().shape, d.cropped_image_path, end='\r')
        i += 1
        # print(d.getImage().shape)
        # print(d.getImage().shape)
        # print(d.getImage().shape)


def multiLoad(thread_conf):
    worker_num = thread_conf.thread

    total_task = len(train_data)
    jobs = []
    task_per_worker = math.ceil(total_task / worker_num)
    st_idx = [task_per_worker * i for i in range(worker_num)]
    ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(worker_num)]
    for i in range(worker_num):
        # temp_data_processor = copy.deepcopy(data_processor)
        p = multiprocessing.Process(target=loadImageData, args=(
            train_data[st_idx[i]:ed_idx[i]],))
        jobs.append(p)
        p.start()
        print('errorfile check start ', i)
    print('all start')
    for p in jobs:
        p.join()
    print('all end')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model arguments')

    parser.add_argument('-td', '--trainDataDir', nargs='+', type=str, help='training image directories')
    parser.add_argument('-t', '--thread', default='16', type=int,
                        help='thread number for multiprocessing')

    run_args = parser.parse_args()

    for dir_x in run_args.trainDataDir:
        addFlushData(dir_x)
    while True:
        multiLoad(run_args)
