import numpy as np
from skimage import io, transform
import os
import sys
import matplotlib.pyplot as plt
import math
import time
import argparse
import ast
import scipy.io as sio
import copy
from visualize import show, showMesh, showImage, showLandmark, showLandmark2
import pickle
from torchdata import ImageData
from torchmodel import TorchNet
from torchdata import getDataLoader
from torchloss import getErrorFunction, getLossFunction
import torch
import tensorboard


class NetworkManager:
    def __init__(self, args):
        self.train_data = []
        self.val_data = []
        self.test_data = []

        self.gpu_num = args.gpu
        self.batch_size = args.batchSize
        self.model_save_path = args.modelSavePath
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.epoch = args.epoch
        self.start_epoch = args.startEpoch

        self.error_function = args.errorFunction

        self.net = TorchNet(gpu_num=args.gpu, visible_gpus=args.visibleDevice, learning_rate=args.learningRate)  # class of
        # RZYNet
        # if true, provide [pos offset R T] as groundtruth. Otherwise ,provide pos as GT

        # 0: normal PRN [image posmap]  1: offset [image offset R T S]
        self.data_mode = 0
        self.weight_decay = 0.0001

        self.criterion = None
        self.metrics = None

    def buildModel(self, args):
        print('building', args.netStructure)
        if args.netStructure == 'InitPRN':
            self.data_mode = 0
            self.net.buildInitPRN()

        elif args.netStructure == 'OffsetPRN':
            self.data_mode = 1
            self.net.buildOffsetPRN()

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
        print(len(all_data), 'data added')

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

    def saveImageDataPaths(self, save_folder='data'):
        print('saving data path list')
        ft = open(save_folder + '/' + 'train_data.pkl', 'wb')
        fv = open(save_folder + '/' + 'val_data.pkl', 'wb')
        pickle.dump(self.train_data, ft)
        pickle.dump(self.val_data, fv)
        ft.close()
        fv.close()
        print('data path list saved')

    def loadImageDataPaths(self, load_folder='data'):
        print('loading data path list')
        ft = open(load_folder + '/' + 'train_data.pkl', 'rb')
        fv = open(load_folder + '/' + 'val_data.pkl', 'rb')
        self.train_data = pickle.load(ft)
        self.val_data = pickle.load(fv)
        ft.close()
        fv.close()
        print('data path list loaded')

    def train(self):
        best_acc = 1000
        model = self.net.model
        optimizer = self.net.optimizer
        scheduler = self.net.scheduler

        from thop import profile
        sample_input = torch.randn((1, 3, 256, 256)).to(self.net.device)
        sample_output = torch.randn((1, 3, 256, 256)).to(self.net.device)
        flops, params = profile(model, inputs=(sample_input, sample_output))
        print('params:%d  flops:%d' % (params, flops))

        # l2_weight_loss = torch.tensor(0).to(self.net.device).float()
        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         l2_weight_loss += torch.norm(param, 2)

        if self.data_mode == 1:
            train_data_loader = getDataLoader(self.train_data, mode='offset', batch_size=self.batch_size * self.gpu_num, is_shuffle=True, is_aug=True)
            val_data_loader = getDataLoader(self.val_data, mode='offset', batch_size=self.batch_size * self.gpu_num, is_shuffle=False, is_aug=False)
        else:
            train_data_loader = getDataLoader(self.train_data, mode='posmap', batch_size=self.batch_size * self.gpu_num, is_shuffle=True, is_aug=True)
            val_data_loader = getDataLoader(self.val_data, mode='posmap', batch_size=self.batch_size * self.gpu_num, is_shuffle=False, is_aug=False)

        for epoch in range(self.start_epoch, self.epoch):
            print('Epoch: %d' % epoch)
            scheduler.step()
            model.train()
            total_itr_num = len(train_data_loader.dataset) // train_data_loader.batch_size

            sum_loss = 0.0
            t_start = time.time()
            if self.data_mode == 1:
                sum_metric_loss = np.zeros(5)
            else:
                sum_metric_loss = 0.0
            for i, data in enumerate(train_data_loader):
                # 准备数据
                if self.data_mode == 1:
                    x = data[0]
                    x = x.to(self.net.device).float()
                    y = [data[j] for j in range(1, 6)]
                    for j in range(5):
                        y[j] = y[j].to(self.net.device).float()
                    optimizer.zero_grad()
                    outputs = model(x, y[0], y[1], y[2], y[3], y[4])
                    loss = torch.mean(outputs[0])
                    metrics_loss = [torch.mean(outputs[j]) for j in range(1, 6)]
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    print('\r', end='')
                    print('[epoch:%d, iter:%d/%d, time:%d] Loss: %.04f ' % (epoch, i + 1, total_itr_num, int(time.time() - t_start), sum_loss / (i + 1)),
                          end='')
                    for j in range(5):
                        sum_metric_loss[j] += metrics_loss[j]
                        print(' Metrics%d: %.04f ' % (j, sum_metric_loss[j] / (i + 1)), end='')

                else:  # datamode==0
                    x = data[0]
                    y = data[1]
                    x, y = x.to(self.net.device).float(), y.to(self.net.device).float()
                    optimizer.zero_grad()
                    outputs = model(x, y)
                    loss = outputs[0]
                    metrics_loss = outputs[1]
                    loss = torch.mean(loss)
                    metrics_loss = torch.mean(metrics_loss)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    sum_metric_loss += metrics_loss.item()
                    print('\r[epoch:%d, iter:%d/%d, time:%d] Loss: %.04f  Metrics: %.04f'
                          % (epoch, i + 1, total_itr_num, int(time.time() - t_start), sum_loss / (i + 1), sum_metric_loss / (i + 1)), end='')

            # validation
            print("\nWaiting Test!", end='\r')
            with torch.no_grad():
                if self.data_mode == 1:
                    sum_metric_loss = np.zeros(5)
                else:
                    sum_metric_loss = 0.0
                for data in val_data_loader:
                    model.eval()
                    if self.data_mode == 1:
                        x = data[0]
                        x = x.to(self.net.device).float()
                        y = [data[j] for j in range(1, 6)]
                        for j in range(5):
                            y[j] = y[j].to(self.net.device).float()
                        outputs = model(x, y[0], y[1], y[2], y[3], y[4])
                        metrics_loss = [torch.mean(outputs[j]) for j in range(1, 6)]
                        for j in range(5):
                            sum_metric_loss[j] += metrics_loss[j]

                    else:  # datamode==0
                        x = data[0]
                        y = data[1]
                        x, y = x.to(self.net.device), y.to(self.net.device)
                        x = x.float()
                        y = y.float()
                        outputs = model(x, y)
                        metrics_loss = outputs[1]
                        metrics_loss = torch.mean(metrics_loss)
                        sum_metric_loss += metrics_loss

                if self.data_mode == 1:
                    for j in range(5):
                        print('val Metrics%d: %.04f ' % (j, sum_metric_loss[j] / len(val_data_loader)), end='')
                    sum_metric_loss = sum_metric_loss[0]
                else:
                    print('val metrics: %.4f' % (sum_metric_loss / len(val_data_loader)), end='')

                print('\nSaving model......', end='\r')
                if self.gpu_num > 1:
                    torch.save(model.module.state_dict(), '%s/net_%03d.pth' % (self.model_save_path, epoch + 1))
                else:
                    torch.save(model.state_dict(), '%s/net_%03d.pth' % (self.model_save_path, epoch + 1))
                # save best
                if sum_metric_loss / len(val_data_loader) < best_acc:
                    print('new best %.4f improved from %.4f' % (sum_metric_loss / len(val_data_loader), best_acc))
                    best_acc = sum_metric_loss / len(val_data_loader)
                    if self.gpu_num > 1:
                        torch.save(model.module.state_dict(), '%s/best.pth' % self.model_save_path)
                    else:
                        torch.save(model.state_dict(), '%s/best.pth' % self.model_save_path)
                else:
                    print('not improved from %.4f' % best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model arguments')

    parser.add_argument('--gpu', default=1, type=int, help='gpu number')
    parser.add_argument('--batchSize', default=16, type=int, help='batchsize')
    parser.add_argument('--epoch', default=30, type=int, help='epoch')
    parser.add_argument('--modelSavePath', default='savedmodel/temp_best_model', type=str, help='model save path')
    parser.add_argument('-td', '--trainDataDir', nargs='+', type=str, help='training image directories')
    parser.add_argument('-vd', '--valDataDir', nargs='+', type=str, help='validation image directories')
    parser.add_argument('-pd', '--testDataDir', nargs='+', type=str, help='test/predict image directories')
    parser.add_argument('--foreFaceMaskPath', default='uv-data/uv_face_mask.png', type=str, help='')
    parser.add_argument('--weightMaskPath', default='uv-data/uv_weight_mask.png', type=str, help='')
    parser.add_argument('--uvKptPath', default='uv-data/uv_kpt_ind.txt', type=str, help='')
    parser.add_argument('-train', '--isTrain', default=False, type=ast.literal_eval, help='')
    parser.add_argument('-test', '--isTest', default=False, type=ast.literal_eval, help='')
    parser.add_argument('-testsingle', '--isTestSingle', default=False, type=ast.literal_eval, help='')
    parser.add_argument('-visualize', '--isVisualize', default=False, type=ast.literal_eval, help='')
    parser.add_argument('--errorFunction', default='nme2d', nargs='+', type=str)
    parser.add_argument('--loadModelPath', default=None, type=str, help='')
    parser.add_argument('--visibleDevice', default='0', type=str, help='')
    parser.add_argument('-struct', '--netStructure', default='InitPRNet', type=str, help='')
    parser.add_argument('-lr', '--learningRate', default=1e-4, type=float)
    parser.add_argument('--startEpoch', default=0, type=int)

    run_args = parser.parse_args()

    print(run_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = run_args.visibleDevice
    print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name(0))

    net_manager = NetworkManager(run_args)
    net_manager.buildModel(run_args)
    if run_args.isTrain:
        if run_args.trainDataDir is not None:
            if run_args.valDataDir is not None:
                for dir in run_args.trainDataDir:
                    net_manager.addImageData(dir, 'train')
                for dir in run_args.valDataDir:
                    net_manager.addImageData(dir, 'val')
            else:
                for dir in run_args.trainDataDir:
                    net_manager.addImageData(dir, 'both')
            net_manager.saveImageDataPaths()
        else:
            net_manager.loadImageDataPaths()

        if run_args.loadModelPath is not None:
            net_manager.net.loadWeights(run_args.loadModelPath)
        net_manager.train()
