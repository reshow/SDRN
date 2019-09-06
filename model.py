import keras
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Conv2DTranspose
from keras.layers import Add, GlobalMaxPooling2D
from keras.layers import add, Flatten
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, Callback, History
from keras.optimizers import adam
from keras import backend as K
from keras.utils import multi_gpu_model
from skimage import io, transform
import os
import matplotlib.pyplot as plt
import math
from data import ImageData, FitGenerator, DataProcessor
import time
import argparse
import ast
import scipy.io as sio
import copy

data_processor = DataProcessor()


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def ResBlock(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


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


class RZYModel:
    class Config:
        def __init__(self,
                     gpu_num,
                     batch_size,
                     model_save_path,
                     paral_model_save_path,
                     epoch,
                     loss_function,
                     optimizer,
                     foreface_path,
                     weight_path,
                     uv_kpt_path,
                     train_data_dirs,
                     val_data_dirs,
                     test_dirs,
                     error_function
                     ):
            self.gpu_num = gpu_num
            self.batch_size = batch_size
            self.model_save_path = model_save_path
            self.paral_model_save_path = paral_model_save_path
            self.epoch = epoch
            self.loss_function = loss_function
            self.optimizer = optimizer
            self.foreface_path = foreface_path
            self.weight_path = weight_path
            self.uv_kpt_path = uv_kpt_path
            self.train_data_dirs = train_data_dirs
            self.val_data_dirs = val_data_dirs
            self.test_dirs = test_dirs
            self.error_function = error_function

    class LossLib:
        def __init__(self, is_foreface=True, foreface_path='uv-data/uv_face_mask.png',
                     is_weighted=True, weight_path='uv-data/uv_weight_mask.png',
                     is_kpt=True, uv_kpt_path='uv-data/uv_kpt_ind.txt'):
            if is_foreface:
                face_mask = io.imread(foreface_path) / 255.
                # face_mask = np.reshape(face_mask, (256, 256, 1))
                # face_mask = np.concatenate([face_mask] * 3, 2)
                [image_h, image_w] = face_mask.shape
                self.face_mask_mean_fix_rate = (image_h * image_w) / np.sum(face_mask)
                self.face_mask_np = copy.copy(face_mask)
                face_mask = K.variable(face_mask)
                self.face_mask = face_mask
                # should divide 43867 foreface vertices but not 65536 total pixels to calculate mean error
            if is_weighted:
                weight_mask = io.imread(weight_path) / 255.
                weight_mask = K.variable(weight_mask)
                self.weight_mask = weight_mask
            if is_kpt:
                file = open(uv_kpt_path, 'r', encoding='utf-8')
                lines = file.readlines()
                # txt is inversed
                x_line = lines[1]
                y_line = lines[0]
                self.uv_kpt = np.zeros((68, 2)).astype(int)
                x_tokens = x_line.strip().split(' ')
                y_tokens = y_line.strip().split(' ')
                for i in range(68):
                    self.uv_kpt[i][0] = int(float(x_tokens[i]))
                    self.uv_kpt[i][1] = int(float(y_tokens[i]))

        def PRNLoss(self, is_foreface=False, is_weighted=False):
            """
            here is a tricky way to customize loss functions for keras
            :param is_foreface:
            :param is_weighted:
            :return: loss function in keras format
            """

            def templateLoss(y_true, y_pred):
                dist = K.sqrt(K.sum(K.square(K.abs(y_true - y_pred)), axis=-1))
                if is_weighted:
                    dist = dist * self.weight_mask
                if is_foreface:
                    dist = dist * self.face_mask * self.face_mask_mean_fix_rate
                loss = K.mean(dist)
                return loss

            return templateLoss

        def getLossFunction(self, loss_func_name='SquareError'):
            if loss_func_name == 'RootSquareError' or loss_func_name == 'rse':
                return self.PRNLoss(is_foreface=False, is_weighted=False)
            elif loss_func_name == 'WeightedRootSquareError' or loss_func_name == 'wrse':
                return self.PRNLoss(is_foreface=False, is_weighted=True)
            elif loss_func_name == 'ForefaceRootSquareError' or loss_func_name == 'frse':
                return self.PRNLoss(is_foreface=True, is_weighted=False)
            elif loss_func_name == 'ForefaceWeightedRootSquareError' or loss_func_name == 'fwrse':
                return self.PRNLoss(is_foreface=True, is_weighted=True)
            else:
                print('unknown loss:', loss_func_name)

        def PRNError(self, is_2d=False, is_normalized=True, is_foreface=True, is_landmark=False, is_gt_landmark=False):
            def templateError(y_true, y_pred, bbox=None, landmarks=None):
                assert (not (is_foreface and is_landmark))
                if is_landmark:
                    # the gt landmark is not the same as the landmarks get from mesh using index
                    if is_gt_landmark:
                        gt = landmarks
                        pred = y_pred[self.uv_kpt[:, 0], self.uv_kpt[:, 1]]
                        diff = np.square(gt - pred)
                        if is_2d:
                            dist = np.sqrt(np.sum(diff[:, 0:2], axis=-1))
                        else:
                            dist = np.sqrt(np.sum(diff, axis=-1))
                    else:
                        gt = y_true[self.uv_kpt[:, 0], self.uv_kpt[:, 1]]
                        pred = y_pred[self.uv_kpt[:, 0], self.uv_kpt[:, 1]]
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
                        dist = dist * self.face_mask_np * self.face_mask_mean_fix_rate

                if is_normalized:
                    # bbox_size = np.sqrt(np.sum(np.square(bbox[0, :] - bbox[1, :])))
                    bbox_size = np.sqrt((bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1]))
                else:
                    bbox_size = 1.
                loss = np.mean(dist / bbox_size)
                return loss

            return templateError

        def getErrorFunction(self, error_func_name='NME'):
            if error_func_name == 'nme2d' or error_func_name == 'normalized mean error2d':
                return self.PRNError(is_2d=True, is_normalized=True, is_foreface=True)
            elif error_func_name == 'nme3d' or error_func_name == 'normalized mean error3d':
                return self.PRNError(is_2d=False, is_normalized=True, is_foreface=True)
            elif error_func_name == 'landmark2d' or error_func_name == 'normalized mean error3d':
                return self.PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True)
            elif error_func_name == 'landmark3d' or error_func_name == 'normalized mean error3d':
                return self.PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True)
            elif error_func_name == 'gtlandmark2d' or error_func_name == 'normalized mean error3d':
                return self.PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True,
                                     is_gt_landmark=True)
            elif error_func_name == 'gtlandmark3d' or error_func_name == 'normalized mean error3d':
                return self.PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True,
                                     is_gt_landmark=True)
            else:
                print('unknown error:', error_func_name)

    def __init__(self, args):
        self.conf = self.Config(gpu_num=args.gpu,
                                batch_size=args.batchSize,
                                model_save_path=args.modelSavePath,
                                paral_model_save_path=args.paralModelSavePath,
                                epoch=args.epoch,
                                loss_function=args.lossFunction,
                                optimizer=args.optimizer,
                                foreface_path=args.foreFaceMaskPath,
                                weight_path=args.weightMaskPath,
                                uv_kpt_path=args.uvKptPath,
                                train_data_dirs=args.trainDataDir,
                                val_data_dirs=args.valDataDir,
                                test_dirs=args.testDataDir,
                                error_function=args.errorFunction)
        self.loss_lib = self.LossLib(foreface_path=self.conf.foreface_path, weight_path=self.conf.weight_path,
                                     uv_kpt_path=self.conf.uv_kpt_path)
        self.model = None
        self.paral_model = None
        self.train_data = []
        self.val_data = []
        self.test_data = []
        pass

    def buildPRNet(self):
        feature_size = 16
        inpt = Input(shape=(256, 256, 3,))
        x = Conv2d_BN(inpt, nb_filter=feature_size, kernel_size=4, strides=(1, 1), padding='same')  # 256 256 16
        # x = Conv2D(filter=size, kernel_size=, padding='same', strides=(1,1), activation='relu', name='Conv0')(inpt)
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(2, 2),
                     with_conv_shortcut=True)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 2, kernel_size=4, strides=(1, 1),
                     with_conv_shortcut=False)  # 128 128 32
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 64 64 64
        x = ResBlock(x, nb_filter=feature_size * 4, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 64 64 64
        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 8, kernel_size=4, strides=(1, 1),
                     with_conv_shortcut=False)  # 32 32 128
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(2, 2),
                     with_conv_shortcut=True)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 16, kernel_size=4, strides=(1, 1),
                     with_conv_shortcut=False)  # 16 16 256
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(2, 2), with_conv_shortcut=True)  # 8 8 512
        x = ResBlock(x, nb_filter=feature_size * 32, kernel_size=4, strides=(1, 1), with_conv_shortcut=False)  # 8 8 512

        x = Conv2DTranspose(filters=feature_size * 32, kernel_size=4, strides=(1, 1), activation='relu',
                            padding='same')(
            x)  # 8 8 512
        x = Conv2DTranspose(filters=feature_size * 16, kernel_size=4, strides=(2, 2), activation='relu',
                            padding='same')(
            x)  # 16 16 256
        x = Conv2DTranspose(filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu',
                            padding='same')(
            x)  # 16 16 256
        x = Conv2DTranspose(filters=feature_size * 16, kernel_size=4, strides=(1, 1), activation='relu',
                            padding='same')(
            x)  # 16 16 256
        x = Conv2DTranspose(filters=feature_size * 8, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 32 32 128
        x = Conv2DTranspose(filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 32 32 128
        x = Conv2DTranspose(filters=feature_size * 8, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 32 32 128
        x = Conv2DTranspose(filters=feature_size * 4, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 64 64 64
        x = Conv2DTranspose(filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 64 64 64
        x = Conv2DTranspose(filters=feature_size * 4, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 64 64 64

        x = Conv2DTranspose(filters=feature_size * 2, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 128 128 32
        x = Conv2DTranspose(filters=feature_size * 2, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 128 128 32
        x = Conv2DTranspose(filters=feature_size, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(
            x)  # 256 256 16
        x = Conv2DTranspose(filters=feature_size, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(
            x)  # 256 256 16

        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(x)  # 256 256 3
        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='relu', padding='same')(x)  # 256 256 3
        x = Conv2DTranspose(filters=3, kernel_size=4, strides=(1, 1), activation='sigmoid', padding='same')(
            x)  # 256 256 3

        model = Model(inputs=inpt, outputs=x)
        self.model = model

        if self.conf.gpu_num > 1:
            para_model = multi_gpu_model(model, gpus=self.conf.gpu_num)
            para_model.compile(loss=self.loss_lib.getLossFunction(self.conf.loss_function),
                               optimizer=self.conf.optimizer,
                               metrics=[self.loss_lib.getLossFunction('frse'), 'mae'])
            para_model.summary()
            self.paral_model = para_model
        else:
            model.compile(loss=self.loss_lib.getLossFunction(self.conf.loss_function),
                          optimizer=self.conf.optimizer,
                          metrics=[self.loss_lib.getLossFunction('frse'), 'mae'])
            model.summary()
            self.model = model

    def loadWeights(self, model_weights_path):
        self.model.load_weights(model_weights_path)

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
        checkpointer = ParallelModelCheckpoint(self.model, filepath=self.conf.model_save_path, monitor='loss',
                                               verbose=1,
                                               save_best_only=True, save_weights_only=True)
        train_gen = FitGenerator(self.train_data)
        val_gen = FitGenerator(self.val_data)
        tensorboard_dir = 'tmp' + '/' + str(int(time.time()))
        print('number of data images:', len(self.train_data), len(self.val_data))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_images=1, histogram_freq=0)
        if self.conf.gpu_num > 1:
            self.paral_model.fit_generator(train_gen.gen(batch_size=self.conf.batch_size, gen_mode='order'),
                                           steps_per_epoch=math.ceil(
                                               len(self.train_data) / float(self.conf.batch_size)),
                                           epochs=self.conf.epoch,
                                           verbose=1, callbacks=[checkpointer, tensorboard_callback],
                                           validation_data=val_gen.gen(batch_size=self.conf.batch_size,
                                                                       gen_mode='order'),
                                           validation_steps=math.ceil(len(self.val_data) / float(self.conf.batch_size)))
        else:
            self.model.fit_generator(train_gen.gen(batch_size=self.conf.batch_size, gen_mode='order'),
                                     steps_per_epoch=math.ceil(
                                         len(self.train_data) / float(self.conf.batch_size)),
                                     epochs=self.conf.epoch,
                                     verbose=1, callbacks=[checkpointer, tensorboard_callback],
                                     validation_data=val_gen.gen(batch_size=self.conf.batch_size,
                                                                 gen_mode='order'),
                                     validation_steps=math.ceil(len(self.val_data) / float(self.conf.batch_size)))

    def testBatch(self, index_list, is_init=True, error_func_list=None):
        X = []
        Y = []
        init_X = []
        init_Y = []
        B = []

        # load data and info
        for index in index_list:
            temp_test_data = self.test_data[index]

            cropped_image = io.imread(temp_test_data.x_path) / 255.
            # cropped_image = transform.resize(cropped_image, (256, 256, 3))
            X.append(cropped_image)
            gt = np.load(temp_test_data.y_path)
            Y.append(gt)
            if is_init:
                init_image = io.imread(temp_test_data.init_x_path) / 255.
                init_X.append(init_image)
                init_gt = np.load(temp_test_data.init_y_path)
                init_Y.append(init_gt)

            bbox_info = sio.loadmat(temp_test_data.bbox_info_path)
            B.append(bbox_info)

        X = np.array(X)
        P = self.model.predict(X)
        P = P * 256.  # while training, the posemap has been /256.
        init_P = []

        # visualize
        # for i in range(2):
        #     x = X[i]
        #     p = P[i]
        #     y = Y[i]
        #     texture_image = np.load(self.test_data[index_list[i]].texture_path)
        #     data_processor.show([y, texture_image, x], False, 'uvmap')
        #     data_processor.show([p, texture_image, x], False, 'uvmap')

        if is_init:
            for i in range(len(P)):
                temp_posmap = P[i].copy()
                posmapback = P[i].copy()
                temp_posmap[:, :, 2] = 1
                # posmap = np.dot(posmap, tform.params.T)
                temp_posmap = np.dot(temp_posmap, B[i]['TformInv'].T)
                temp_posmap[:, :, 2] = posmapback[:, :, 2] * B[i]['TformInv'][0][0]
                init_P.append(temp_posmap)

        error_list = []
        for i in range(len(X)):
            temp_errors = []
            for error_func_name in error_func_list:
                error_func = self.loss_lib.getErrorFunction(error_func_name)
                error = error_func(Y[i], P[i], B[i]['Bbox'], B[i]['Kpt'])
                temp_errors.append(error)
                if is_init:
                    init_error = error_func(init_Y[i], init_P[i], B[i]['OldBbox'], B[i]['OldKpt'])
                    temp_errors.append(init_error)
            error_list.append(temp_errors)
        print(np.mean(error_list, axis=0))
        return np.array(error_list)

    def test(self, batch_num=100, error_func_list=None, is_init=True):
        total_task = len(self.test_data)
        print('total img:', total_task)
        task_per_worker = math.ceil(total_task / batch_num)
        st_idx = [task_per_worker * i for i in range(batch_num)]
        ed_idx = [min(total_task, task_per_worker * (i + 1)) for i in range(batch_num)]

        total_error_list = []

        for i in range(batch_num):
            error_list = self.testBatch(np.array(range(st_idx[i], ed_idx[i])), is_init=is_init,
                                        error_func_list=error_func_list)
            total_error_list.extend(error_list)
        total_error_list = np.array(total_error_list)
        mean_errors = np.mean(total_error_list, axis=0)
        for i in range(len(error_func_list)):
            if is_init:
                print(error_func_list[i], mean_errors[i * 2], 'init', mean_errors[i * 2 + 1])
            else:
                print(error_func_list[i], mean_errors[i])

        se_idx = np.argsort(np.sum(total_error_list, axis=-1))
        se_data_list = np.array(self.test_data)[se_idx]
        se_path_list = [a.init_x_path for a in se_data_list]
        sep = '\n'
        fout = open('errororder.txt', 'w', encoding='utf-8')
        fout.write(sep.join(se_path_list))
        fout.close()


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

    parser.add_argument('-l', '--lossFunction', default='fwrse', type=str,
                        help='loss function: rse wrse frse fwrse')
    parser.add_argument('-opt', '--optimizer', default='adam', type=str,
                        help='optimizer: sgd adam')
    parser.add_argument('--foreFaceMaskPath', default='uv-data/uv_face_mask.png', type=str,
                        help='')
    parser.add_argument('--weightMaskPath', default='uv-data/uv_weight_mask.png', type=str,
                        help='')
    parser.add_argument('--uvKptPath', default='uv-data/uv_kpt_ind.txt', type=str,
                        help='')

    parser.add_argument('-train', '--isTrain', default=True, type=ast.literal_eval,
                        help='')
    parser.add_argument('-test', '--isTest', default=True, type=ast.literal_eval,
                        help='')
    parser.add_argument('--errorFunction', default='nme2d', nargs='+', type=str)
    parser.add_argument('--loadModelPath', default=None, type=str,
                        help='')
    parser.add_argument('--visibleDevice', default='0', type=str,
                        help='')
    run_args = parser.parse_args()

    print(run_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = run_args.visibleDevice

    if run_args.isTrain:
        rzy_model = RZYModel(run_args)
        rzy_model.buildPRNet()
        for dir in run_args.trainDataDir:
            rzy_model.addImageData(dir, 'train')
        for dir in run_args.valDataDir:
            rzy_model.addImageData(dir, 'val')
        if run_args.loadModelPath is not None:
            rzy_model.loadWeights(run_args.loadModelPath)
        rzy_model.train()
    if run_args.isTest:
        rzy_model = RZYModel(run_args)
        rzy_model.buildPRNet()
        for dir in run_args.testDataDir:
            rzy_model.addImageData(dir, 'test')
        if run_args.loadModelPath is not None:
            rzy_model.loadWeights(run_args.loadModelPath)
            rzy_model.test(error_func_list=run_args.errorFunction)
