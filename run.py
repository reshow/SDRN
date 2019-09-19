import keras
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, Callback, History, LearningRateScheduler
from keras.optimizers import adam
from keras import backend as K
from skimage import io, transform
import os
import matplotlib.pyplot as plt
import math
from model import RZYNet
import time
import argparse
import ast
import scipy.io as sio
import copy
from loss import getErrorFunction
from data import ImageData, FitGenerator, getLandmark
from visualize import show, showMesh, showImage, showLandmark, showLandmark2
from loss import face_mask_np


class ParallelModelCheckpoint(ModelCheckpoint):
    """
    paralmodel checkpoint for keras
    """

    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


class NetworkManager:
    def __init__(self, args):
        self.train_data = []
        self.val_data = []
        self.test_data = []

        self.gpu_num = args.gpu
        self.batch_size = args.batchSize
        self.model_save_path = args.modelSavePath
        self.paral_model_save_path = args.paralModelSavePath
        self.epoch = args.epoch

        self.error_function = args.errorFunction

        self.net = RZYNet(gpu_num=args.gpu, loss_function=args.lossFunction)  # class of RZYNet
        # if true, provide [pos offset R T] as groundtruth. Otherwise ,provide pos as GT
        self.is_offset_data = False

    def buildModel(self, args):
        print('bulding', args.netStructure)
        if args.netStructure == 'PRNet':
            net_manager.net.buildPRNet()
        elif args.netStructure == 'CbamPRNet':
            net_manager.net.buildCbamPRNet()
        elif args.netStructure == 'InitPRNet':
            net_manager.net.buildInitPRNet()
        elif args.netStructure == 'OffsetPRNet':
            self.is_offset_data = True
            net_manager.net.buildOffsetPRNet()
        else:
            print('unknown network structure')

    def addImageData(self, data_dir, add_mode='train', split_rate=0.8):
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

        if add_mode == 'train':
            self.train_data.extend(all_data)
        elif add_mode == 'val':
            self.val_data.extend(all_data)
        elif add_mode == 'both':
            num_train = math.floor(len(all_data) * split_rate)
            self.train_data.extend(all_data[0:num_train])
            self.val_data.extend(all_data[num_train:])
        elif add_mode == 'test':
            self.test_data.extend(all_data)

    def train(self):
        checkpointer = ParallelModelCheckpoint(self.net.model, filepath=self.model_save_path, monitor='loss',
                                               verbose=1,
                                               save_best_only=True, save_weights_only=True)
        train_gen = FitGenerator(self.train_data)
        val_gen = FitGenerator(self.val_data)
        if self.is_offset_data:
            train_gen_func = train_gen.genOffset(batch_size=self.batch_size * self.gpu_num, gen_mode='random')
            val_gen_func = val_gen.genOffset(batch_size=self.batch_size * self.gpu_num, gen_mode='order')
        else:
            train_gen_func = train_gen.genPRN(batch_size=self.batch_size * self.gpu_num, gen_mode='random')
            val_gen_func = val_gen.genPRN(batch_size=self.batch_size * self.gpu_num, gen_mode='order')

        now_time = time.localtime()
        tensorboard_dir = 'tmp' + '/' + str(now_time.tm_year) + '-' + str(now_time.tm_mon) + '-' + str(now_time.tm_mday) + '-' \
                          + str(now_time.tm_hour) + '-' + str(now_time.tm_min) + '-' + str(now_time.tm_sec)
        print('number of data images:', len(self.train_data), len(self.val_data))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_images=1, histogram_freq=0)

        if self.gpu_num > 1:
            target_model = self.net.paral_model
        else:
            target_model = self.net.model

        def scheduler(epoch):
            # lr decays half every 5 epoch
            if epoch % 5 == 0 and epoch != 0:
                lr = K.get_value(target_model.optimizer.lr)
                K.set_value(target_model.optimizer.lr, lr * 0.5)
                print("lr changed to {}".format(lr * 0.5))
            return K.get_value(target_model.optimizer.lr)

        reduce_lr = LearningRateScheduler(scheduler)
        target_model.fit_generator(train_gen_func,
                                   steps_per_epoch=math.ceil(
                                       len(self.train_data) / float(self.batch_size * self.gpu_num)),
                                   epochs=self.epoch,
                                   verbose=1, callbacks=[checkpointer, tensorboard_callback, reduce_lr],
                                   validation_data=val_gen_func,
                                   validation_steps=math.ceil(len(self.val_data) / float(self.batch_size * self.gpu_num)))

    def test(self, image_data_list, error_func_list=None, is_visualize=False):
        X = []
        Y = []
        B = []
        T = []
        # load data and info
        for temp_test_data in image_data_list:
            cropped_image = io.imread(temp_test_data.cropped_image_path) / 255.
            # cropped_image = transform.resize(cropped_image, (256, 256, 3))
            X.append(cropped_image)
            gt = np.load(temp_test_data.cropped_posmap_path)
            Y.append(gt)
            bbox_info = sio.loadmat(temp_test_data.bbox_info_path)
            B.append(bbox_info)
            if is_visualize:
                texture_image = np.load(temp_test_data.texture_path)
                T.append(texture_image)

        X = np.array(X)
        Y = np.array(Y)
        P = self.net.model.predict(X, batch_size=self.batch_size)
        P = P * 256.  # while training, the posemap has been /256.
        error_list = []
        for i in range(len(X)):
            temp_errors = []
            for error_func_name in error_func_list:
                error_func = getErrorFunction(error_func_name)
                error = error_func(Y[i], P[i], B[i]['Bbox'], B[i]['Kpt'])
                temp_errors.append(error)
            error_list.append(temp_errors)
            if is_visualize:
                print(temp_errors)
                #     data_processor.show([y, texture_image, x], False, 'uvmap')
                #     data_processor.show([p, texture_image, x], False, 'uvmap')
                show([Y[i], T[i], X[i]], False, 'uvmap')
                show([P[i], T[i], X[i]], False, 'uvmap')
                kpt_gt = getLandmark(Y[i])
                kpt_pred = getLandmark(P[i])
                showLandmark2(X[i], kpt_gt, kpt_pred)

                diff = np.square(Y[i] - P[i])
                dist = np.sqrt(np.sum(diff, axis=-1))
                dist = dist * face_mask_np
                plt.imshow(dist)
                plt.show()

        print(np.mean(error_list, axis=0))
        return np.array(error_list)

    def testAllData(self, batch_size=16, error_func_list=None, is_visualize=False):
        total_task = len(self.test_data)
        print('total img:', total_task)
        task_per_worker = batch_size
        itr_num = int(math.ceil(total_task / batch_size))
        st_idx = [task_per_worker * i for i in range(itr_num)]
        ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(itr_num)]

        total_error_list = []

        for i in range(itr_num):
            error_list = self.test(self.test_data[st_idx[i]:ed_idx[i]],
                                   error_func_list=error_func_list, is_visualize=is_visualize)
            total_error_list.extend(error_list)
        total_error_list = np.array(total_error_list)
        mean_errors = np.mean(total_error_list, axis=0)
        for i in range(len(error_func_list)):
            print(error_func_list[i], mean_errors[i])

        se_idx = np.argsort(np.sum(total_error_list, axis=-1))
        se_data_list = np.array(self.test_data)[se_idx]
        se_path_list = [a.cropped_image_path for a in se_data_list]
        sep = '\n'
        fout = open('errororder.txt', 'w', encoding='utf-8')
        fout.write(sep.join(se_path_list))
        fout.close()

    def testSingle(self, image_dir, error_func_list=None, is_visualize=False):
        image_data = ImageData()
        image_data.readPath(image_dir)
        self.test([image_data], error_func_list=error_func_list, is_visualize=is_visualize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model arguments')

    parser.add_argument('--gpu', default=1, type=int,
                        help='gpu number')
    parser.add_argument('--batchSize', default=64, type=int,
                        help='batchsize')
    parser.add_argument('--epoch', default=30, type=int,
                        help='epoch')
    parser.add_argument('--modelSavePath', default='savedmodel/temp_best_model.h5', type=str,
                        help='model save path')
    parser.add_argument('--paralModelSavePath', default='savedmodel/paral_temp_best_model.h5', type=str,
                        help='paralmodel save path')
    parser.add_argument('-td', '--trainDataDir', nargs='+', type=str,
                        help='training image directories')
    parser.add_argument('-vd', '--valDataDir', nargs='+', type=str,
                        help='validation image directories')
    parser.add_argument('-pd', '--testDataDir', nargs='+', type=str,
                        help='test/predict image directories')

    parser.add_argument('-loss', '--lossFunction', default='fwrse', type=str,
                        help='loss function: rse wrse frse fwrse')
    parser.add_argument('--foreFaceMaskPath', default='uv-data/uv_face_mask.png', type=str,
                        help='')
    parser.add_argument('--weightMaskPath', default='uv-data/uv_weight_mask.png', type=str,
                        help='')
    parser.add_argument('--uvKptPath', default='uv-data/uv_kpt_ind.txt', type=str,
                        help='')

    parser.add_argument('-train', '--isTrain', default=False, type=ast.literal_eval,
                        help='')
    parser.add_argument('-test', '--isTest', default=False, type=ast.literal_eval,
                        help='')
    parser.add_argument('-testsingle', '--isTestSingle', default=False, type=ast.literal_eval,
                        help='')
    parser.add_argument('-visualize', '--isVisualize', default=False, type=ast.literal_eval,
                        help='')
    parser.add_argument('--errorFunction', default='nme2d', nargs='+', type=str)
    parser.add_argument('--loadModelPath', default=None, type=str,
                        help='')
    parser.add_argument('--visibleDevice', default='0', type=str,
                        help='')

    parser.add_argument('-struct', '--netStructure', default='InitPRNet', type=str,
                        help='')

    run_args = parser.parse_args()

    print(run_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = run_args.visibleDevice

    net_manager = NetworkManager(run_args)
    net_manager.buildModel(run_args)
    if run_args.isTrain:

        if run_args.valDataDir is not None:
            for dir in run_args.trainDataDir:
                net_manager.addImageData(dir, 'train')
            for dir in run_args.valDataDir:
                net_manager.addImageData(dir, 'val')
        else:
            for dir in run_args.trainDataDir:
                net_manager.addImageData(dir, 'both')
        if run_args.loadModelPath is not None:
            net_manager.net.loadWeights(run_args.loadModelPath)
        net_manager.train()
    if run_args.isTest:
        for dir in run_args.testDataDir:
            net_manager.addImageData(dir, 'test')
        if run_args.loadModelPath is not None:
            net_manager.net.loadWeights(run_args.loadModelPath)
            net_manager.testAllData(error_func_list=run_args.errorFunction, is_visualize=run_args.isVisualize)

    if run_args.isTestSingle:
        net_manager = NetworkManager(run_args)
        net_manager.net.buildPRNet()
        if run_args.loadModelPath is not None:
            net_manager.net.loadWeights(run_args.loadModelPath)
