import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import cv2
import os
import numpy as np

import torchvision
import torchvision.transforms.functional as F
import numbers
import random
from PIL import Image
from torchvision import transforms


class ToTensor(object):
    def __call__(self, sample):
        hazy_image, clean_image = sample['hazy'], sample['clean']
        hazy_image = torch.from_numpy(np.array(hazy_image).astype(np.float32))
        hazy_image = torch.transpose(torch.transpose(hazy_image, 2, 0), 1, 2)
        # hazy_image = hazy_image / 255.0
        clean_image = torch.from_numpy(np.array(clean_image).astype(np.float32))
        clean_image = torch.transpose(torch.transpose(clean_image, 2, 0), 1, 2)
        # clean_image = clean_image / 255.0
        return {'hazy': hazy_image,
                'clean': clean_image}


class Dataset_Load(Dataset):
    def __init__(self, hazy_path, clean_path, transform=None, target_size=(256, 256)):
        self.hazy_dir = hazy_path
        self.clean_dir = clean_path
        self.transform = transform
        self.target_size = target_size  # 设定目标尺寸

    def __len__(self):
        # 假设每个目录下的图像数量相同，这里简单返回720，实际情况可能需要动态获取
        return 720

    def __getitem__(self, index):
        file_list = os.listdir(self.hazy_dir)
        if not file_list:
            raise FileNotFoundError(f"No files found in {self.hazy_dir}")

        name = file_list[index]  # 直接使用索引从列表中获取文件名，避免重复遍历

        hazy_im_path = os.path.join(self.hazy_dir, name)
        clean_im_path = os.path.join(self.clean_dir, name)

        hazy_im = cv2.resize(cv2.imread(hazy_im_path), (256, 256),
                             interpolation=cv2.INTER_AREA)
        if hazy_im is None:
            raise FileNotFoundError(f"File {hazy_im_path} not found")

        hazy_im = hazy_im[:, :, ::-1]  # BGR to RGB
        hazy_im = np.float32(hazy_im) / 255.0

        clean_im = cv2.resize(cv2.imread(clean_im_path), (256, 256),
                             interpolation=cv2.INTER_AREA)
        if clean_im is None:
            raise FileNotFoundError(f"File {clean_im_path} not found")

        clean_im = clean_im[:, :, ::-1]  # BGR to RGB
        clean_im = np.float32(clean_im) / 255.0


        sample = {'hazy': hazy_im,
                  'clean': clean_im}

        if self.transform is not None:
            # 如果self.transform包含额外的变换，现在可以应用它们
            # 注意：这里的self.transform应该不包含Resize，因为它已经被单独应用了
            sample = self.transform(sample)

        return sample

class ValDataset_Load(Dataset):
    def __init__(self, hazy_path, clean_path, transform=None, target_size=(256, 256)):
        self.hazy_dir = hazy_path
        self.clean_dir = clean_path
        self.transform = transform
        self.target_size = target_size  # 设定目标尺寸

    def __len__(self):
        # 假设每个目录下的图像数量相同，这里简单返回720，实际情况可能需要动态获取
        return 80

    def __getitem__(self, index):
        file_list = os.listdir(self.hazy_dir)
        if not file_list:
            raise FileNotFoundError(f"No files found in {self.hazy_dir}")

        name = file_list[index]  # 直接使用索引从列表中获取文件名，避免重复遍历

        hazy_im_path = os.path.join(self.hazy_dir, name)
        clean_im_path = os.path.join(self.clean_dir, name)

        hazy_im = cv2.resize(cv2.imread(hazy_im_path), (256, 256),
                             interpolation=cv2.INTER_AREA)
        if hazy_im is None:
            raise FileNotFoundError(f"File {hazy_im_path} not found")

        hazy_im = hazy_im[:, :, ::-1]  # BGR to RGB
        hazy_im = np.float32(hazy_im) / 255.0

        clean_im = cv2.resize(cv2.imread(clean_im_path), (256, 256),
                             interpolation=cv2.INTER_AREA)
        if clean_im is None:
            raise FileNotFoundError(f"File {clean_im_path} not found")

        clean_im = clean_im[:, :, ::-1]  # BGR to RGB
        clean_im = np.float32(clean_im) / 255.0


        sample = {'hazy': hazy_im,
                  'clean': clean_im}

        if self.transform is not None:
            # 如果self.transform包含额外的变换，现在可以应用它们
            # 注意：这里的self.transform应该不包含Resize，因为它已经被单独应用了
            sample = self.transform(sample)

        return sample
