import numpy as np
import scipy.io as sio
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import augmentation
from PIL import Image
from data import matrix2Angle, matrix2Quaternion, angle2Matrix, angle2Quaternion, quaternion2Matrix, uv_kpt
import os


class ImageData:
    def __init__(self):
        self.cropped_image_path = ''
        self.cropped_posmap_path = ''
        self.init_image_path = ''
        self.init_posmap_path = ''
        self.texture_path = ''
        self.texture_image_path = ''
        self.bbox_info_path = ''
        self.offset_posmap_path = ''
        self.attention_mask_path = ''

        self.image = None
        self.posmap = None
        self.offset_posmap = None
        self.bbox_info = None
        self.S = None
        self.T = None
        self.R = None
        self.attention_mask = None

    def readPath(self, image_dir):
        image_name = image_dir.split('/')[-1]
        self.cropped_image_path = image_dir + '/' + image_name + '_cropped.npy'
        self.cropped_posmap_path = image_dir + '/' + image_name + '_cropped_uv_posmap.npy'
        self.init_image_path = image_dir + '/' + image_name + '_init.jpg'
        self.init_posmap_path = image_dir + '/' + image_name + '_uv_posmap.npy'
        # change the format to npy
        self.texture_path = image_dir + '/' + image_name + '_uv_texture_map.npy'
        self.texture_image_path = image_dir + '/' + image_name + '_uv_texture_map.jpg'

        self.bbox_info_path = image_dir + '/' + image_name + '_bbox_info.mat'
        self.offset_posmap_path = image_dir + '/' + image_name + '_offset_posmap.npy'

        self.attention_mask_path = image_dir + '/' + image_name + '_attention_mask.npy'

    def readFile(self, mode='posmap'):
        if mode == 'posmap':
            self.image = np.load(self.cropped_image_path).astype(np.uint8)
            self.posmap = np.load(self.cropped_posmap_path).astype(np.float16)
        elif mode == 'offset' or mode == 'quaternionoffset':
            self.image = np.load(self.cropped_image_path).astype(np.uint8)
            self.posmap = np.load(self.cropped_posmap_path).astype(np.float16)
            self.offset_posmap = np.load(self.offset_posmap_path).astype(np.float16)
            self.bbox_info = sio.loadmat(self.bbox_info_path)
        elif mode == 'attention':
            self.image = np.load(self.cropped_image_path).astype(np.uint8)
            self.posmap = np.load(self.cropped_posmap_path).astype(np.float16)
            self.attention_mask = np.load(self.attention_mask_path).astype(np.uint8)
        elif mode == 'siam':
            self.image = np.load(self.cropped_image_path).astype(np.uint8)
            self.posmap = np.load(self.cropped_posmap_path).astype(np.float16)
            self.offset_posmap = np.load(self.offset_posmap_path).astype(np.float16)
            self.bbox_info = sio.loadmat(self.bbox_info_path)
        else:
            pass

    def getImage(self):
        if self.image is None:
            return np.load(self.cropped_image_path)
        else:
            return self.image

    def getPosmap(self):
        if self.posmap is None:
            return np.load(self.cropped_posmap_path)
        else:
            return self.posmap

    def getOffsetPosmap(self):
        if self.offset_posmap is None:
            return np.load(self.offset_posmap_path)
        else:
            return self.offset_posmap

    def getBboxInfo(self):
        if self.bbox_info is None:
            return sio.loadmat(self.bbox_info_path)
        else:
            return self.bbox_info

    def getAttentionMask(self):
        if self.attention_mask is None:
            return np.load(self.attention_mask_path)
        else:
            return self.attention_mask


def toTensor(image):
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    return image


class DataGenerator(Dataset):
    def __init__(self, all_image_data, mode='posmap', is_aug=False, is_pre_read=True):
        super(DataGenerator, self).__init__()
        self.all_image_data = all_image_data
        self.image_height = 256
        self.image_width = 256
        self.image_channel = 3
        # mode=posmap or offset
        self.mode = mode
        self.is_aug = is_aug

        self.augment = transforms.Compose(
            [
                transforms.ToPILImage(mode='RGB'),
                transforms.RandomOrder(
                    [transforms.RandomGrayscale(p=0.1),
                     transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.25),
                     # transforms.RandomApply([transforms.Lambda(lambda x: augmentation.channelScale(x))], p=0.25),
                     # transforms.RandomApply([transforms.Lambda(lambda x: augmentation.randomErase(x))], p=0.25)
                     ]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        self.toTensor = transforms.ToTensor()
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.no_augment = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.is_pre_read = is_pre_read
        if is_pre_read:
            i = 0
            print('preloading')
            if self.mode == 'posmap':
                num_max_PR = 80000
            else:
                num_max_PR = 40000
            for data in self.all_image_data:
                data.readFile(mode=self.mode)
                print(i, end='\r')
                i += 1
                if i > num_max_PR:
                    break

    def __getitem__(self, index):
        if self.mode == 'posmap':

            image = (self.all_image_data[index].getImage() / 255.0).astype(np.float32)
            pos = self.all_image_data[index].getPosmap().astype(np.float32)
            if self.is_aug:
                image, pos = augmentation.prnAugment_torch(image, pos)
                # image = (image * 255.0).astype(np.uint8)
                #  image = self.augment(image)

                # image = self.no_augment(image)

                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            else:
                # image = (image - image.mean()) / np.sqrt(image.var() + 0.001)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
                # image = self.no_augment(image)
            pos = pos / 280.
            pos = self.toTensor(pos)
            return image, pos

        elif self.mode == 'offset':

            image = (self.all_image_data[index].getImage() / 255.).astype(np.float32)
            pos = self.all_image_data[index].getPosmap().astype(np.float32)
            offset = self.all_image_data[index].getOffsetPosmap().astype(np.float32)
            bbox_info = self.all_image_data[index].getBboxInfo()
            trans_mat = bbox_info['TformOffset']

            if self.is_aug:
                if np.random.rand() > 0.5:
                    rot_angle = np.random.randint(-90, 90)
                    rot_angle = rot_angle / 180. * np.pi
                    R_3d, R_3d_inv = augmentation.getRotateMatrix3D(rot_angle, image.shape)
                    trans_mat = R_3d.dot(trans_mat)
                    image, pos = augmentation.rotateData(image, pos, specify_angle=rot_angle)
                image, pos = augmentation.prnAugment_torch(image, pos, is_rotate=False)
                # image = (image * 255.0).astype(np.uint8)
                # image = self.augment(image)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            else:
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)

            t0 = trans_mat[0:3, 0]
            S = np.sqrt(np.sum(t0 * t0))
            R = trans_mat[0:3, 0:3]
            R = R.dot(np.diagflat([1 / S, -1 / S, 1 / S]))
            R_flatten = matrix2Angle(R)
            R_flatten = np.reshape((np.array(R_flatten)), (3,)) / np.pi
            for i in range(3):
                while R_flatten[i] < -1:
                    R_flatten[i] += 2
                while R_flatten[i] > 1:
                    R_flatten[i] -= 2
            # R_flatten = np.reshape(R, (9,))

            T_flatten = np.reshape(trans_mat[0:3, 3], (3,))
            S = S * 5e2
            T_flatten = T_flatten / 300

            if S > 1:
                print('too large scale', S)
            if (abs(T_flatten) > 1).any():
                print('too large T', T_flatten)
            if (abs(R_flatten) > 1).any():
                print('too large R', R_flatten)

            R_flatten = torch.from_numpy(R_flatten)
            T_flatten = torch.from_numpy(T_flatten)
            S = torch.tensor(S)

            pos = pos / 280.
            offset = offset / 4.
            pos = self.toTensor(pos)
            offset = self.toTensor(offset)

            return image, pos, offset, R_flatten, T_flatten, S

        elif self.mode == 'attention':
            image = (self.all_image_data[index].getImage() / 255.0).astype(np.float32)
            pos = self.all_image_data[index].getPosmap().astype(np.float32)
            attention_mask = self.all_image_data[index].getAttentionMask().astype(np.float32)
            if self.is_aug:
                image, pos, attention_mask = augmentation.attentionAugment_torch(image, pos, attention_mask)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            else:
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            pos = pos / 280.
            pos = self.toTensor(pos)
            attention_mask = Image.fromarray(attention_mask)
            attention_mask = attention_mask.resize((32, 32), Image.BILINEAR)
            attention_mask = np.array(attention_mask)
            attention_mask = self.toTensor(attention_mask)
            return image, pos, attention_mask

        elif self.mode == 'quaternionoffset' or self.mode == 'meanoffset':
            image = (self.all_image_data[index].getImage() / 255.).astype(np.float32)
            pos = self.all_image_data[index].getPosmap().astype(np.float32)
            offset = self.all_image_data[index].getOffsetPosmap().astype(np.float32)
            bbox_info = self.all_image_data[index].getBboxInfo()
            trans_mat = bbox_info['TformOffset']

            if self.is_aug:
                if np.random.rand() > 0.5:
                    rot_angle = np.random.randint(-90, 90)
                    rot_angle = rot_angle / 180. * np.pi
                    R_3d, R_3d_inv = augmentation.getRotateMatrix3D(rot_angle, image.shape)
                    trans_mat = R_3d.dot(trans_mat)
                    image, pos = augmentation.rotateData(image, pos, specify_angle=rot_angle)
                image, pos = augmentation.prnAugment_torch(image, pos, is_rotate=False)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            else:
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)

            t0 = trans_mat[0:3, 0]
            S = np.sqrt(np.sum(t0 * t0))
            R = trans_mat[0:3, 0:3]
            R = R.dot(np.diagflat([1 / S, -1 / S, 1 / S]))

            Q = matrix2Quaternion(R)
            Qf = np.reshape(np.array(Q) * np.sqrt(S), (4,))

            T_flatten = np.reshape(trans_mat[0:3, 3], (3,))
            Qf = Qf * 20
            T_flatten = T_flatten / 300

            # print(Qf.max(),Qf.min(),T_flatten.max(),T_flatten.min())

            if (abs(Qf) > 1).any():
                print('too large Q', Qf)

            Qf = torch.from_numpy(Qf)
            T_flatten = torch.from_numpy(T_flatten)

            pos = pos / 280.
            offset = offset / 4.
            pos = self.toTensor(pos)
            offset = self.toTensor(offset)
            return image, pos, offset, Qf, T_flatten,

        elif self.mode == 'siam':

            image = (self.all_image_data[index].getImage() / 255.).astype(np.float32)
            pos = self.all_image_data[index].getPosmap().astype(np.float32)
            offset = self.all_image_data[index].getOffsetPosmap().astype(np.float32)

            if self.is_aug:
                # if np.random.rand() > 0.5:
                #     rot_angle = np.random.randint(-90, 90)
                #     rot_angle = rot_angle / 180. * np.pi
                #     image, pos = augmentation.rotateData(image, pos, specify_angle=rot_angle)
                # image, pos = augmentation.prnAugment_torch(image, pos, is_rotate=False)
                image, pos = augmentation.prnAugment_torch(image, pos)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            else:
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)

            pos = pos / 280.
            offset = offset / 6.
            if abs(offset).max() > 1:
                print('\n too large offset', abs(offset).max())
            pos = self.toTensor(pos)
            offset = self.toTensor(offset)
            return image, pos, offset

        elif self.mode == 'visible':

            image = (self.all_image_data[index].getImage() / 255.).astype(np.float32)
            pos = self.all_image_data[index].getPosmap().astype(np.float32)
            offset = self.all_image_data[index].getOffsetPosmap().astype(np.float32)
            attention_mask = self.all_image_data[index].getAttentionMask().astype(np.float32)

            if self.is_aug:
                # if np.random.rand() > 0.5:
                #     rot_angle = np.random.randint(-90, 90)
                #     rot_angle = rot_angle / 180. * np.pi
                #     image, pos = augmentation.rotateData(image, pos, specify_angle=rot_angle)
                # image, pos = augmentation.prnAugment_torch(image, pos, is_rotate=False)
                image, pos, attention_mask = augmentation.attentionAugment_torch(image, pos, attention_mask)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            else:
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)

            pos = pos / 280.
            offset = offset / 6.
            if abs(offset).max() > 1:
                print('\n too large offset', abs(offset).max())
            pos = self.toTensor(pos)
            offset = self.toTensor(offset)
            attention_mask = Image.fromarray(attention_mask)
            attention_mask = attention_mask.resize((32, 32), Image.BILINEAR)
            attention_mask = np.array(attention_mask)
            attention_mask = self.toTensor(attention_mask)
            return image, pos, offset, attention_mask

        elif self.mode == 'kpt':
            if os.path.exists(self.all_image_data[index].cropped_posmap_path):
                image = (self.all_image_data[index].getImage() / 255.).astype(np.float32)
                pos = self.all_image_data[index].getPosmap().astype(np.float32)
                offset = self.all_image_data[index].getOffsetPosmap().astype(np.float32)
                attention_mask = self.all_image_data[index].getAttentionMask().astype(np.float32)

                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)

                pos = pos / 280.
                offset = offset / 6.
                if abs(offset).max() > 1:
                    print('\n too large offset', abs(offset).max())
                pos = self.toTensor(pos)
                offset = self.toTensor(offset)
                attention_mask = Image.fromarray(attention_mask)
                attention_mask = attention_mask.resize((32, 32), Image.BILINEAR)
                attention_mask = np.array(attention_mask)
                attention_mask = self.toTensor(attention_mask)
                return image, pos, offset, attention_mask
            else:
                image = (self.all_image_data[index].getImage() / 255.).astype(np.float32)
                offset = np.zeros((256, 256, 3)).astype(np.float32)

                attention_mask = np.zeros((256, 256)).astype(np.float32)
                bbox_info = self.all_image_data[index].getBboxInfo()
                kpt = bbox_info['Kpt'].astype(np.float32)

                if self.is_aug:
                    # if np.random.rand() > 0.5:
                    #     rot_angle = np.random.randint(-90, 90)
                    #     rot_angle = rot_angle / 180. * np.pi
                    #     image, pos = augmentation.rotateData(image, pos, specify_angle=rot_angle)
                    # image, pos = augmentation.prnAugment_torch(image, pos, is_rotate=False)
                    image, kpt, attention_mask = augmentation.kptAugment(image, kpt, attention_mask)
                    for i in range(3):
                        image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                    image = self.toTensor(image)
                else:
                    for i in range(3):
                        image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                    image = self.toTensor(image)

                attention_mask = np.zeros((32, 32)).astype(np.float32)
                kpt = (kpt.transpose() / 280.).astype(np.float32)
                offset = offset / 6.
                if abs(offset).max() > 1:
                    print('\n too large offset', abs(offset).max())
                kpt = torch.from_numpy(kpt)
                offset = self.toTensor(offset)
                attention_mask = self.toTensor(attention_mask)
                return image, kpt, offset, attention_mask
        else:
            return None

    def __len__(self):
        return len(self.all_image_data)


def getDataLoader(all_image_data, mode='posmap', batch_size=16, is_shuffle=False, is_aug=False, is_pre_read=True, num_worker=8):
    dataset = DataGenerator(all_image_data=all_image_data, mode=mode, is_aug=is_aug, is_pre_read=is_pre_read)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_worker, pin_memory=False, drop_last=True)
    return train_loader
