# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 19:29
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : ms_ssim.py
# @Software: PyCharm


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import random

import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average=True, max_val=255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val

    def _ssim(self, img1, img2, size_average = True):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=self.channel) - mu1_mu2

        C1 = (0.01 * self.max_val)**2
        C2 = (0.03 * self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):
        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels,).cuda())
        mcs = Variable(torch.Tensor(levels,).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = torch.relu(ssim_map)
            mcs[i] = torch.relu(mcs_map)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1]) * (msssim[levels-1]**weight[levels-1]))
        return value

    def forward(self, img1, img2):
        return self.ms_ssim(img1, img2)


def load_image(file_path, input_height=128, input_width=None, output_height=128, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False):
    '''
    读取图像，是否做增强
    '''
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height

    img = Image.open(file_path)
    if is_gray is False and img.mode is not 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode is not 'L':
        img = img.convert('L')

    # 随机进行水平翻转
    if is_mirror and np.random.randint(0, 1) is 0:
        img = ImageOps.mirror(img)

    if input_height is not None:
        img = img.resize((input_width, input_height), Image.BICUBIC)

    # 去掉左，上，右，下四个边上的行/列数
    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
            #print([w,cropSize])
            cx1 = np.random.randint(0, w - crop_width)
            cx2 = w - crop_width - cx1
            cy1 = np.random.randint(0, h - crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w-crop_width)/2.))
            cy2 = cy1 = int(round((h-crop_height)/2.))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))

    img = img.resize((output_width, output_height), Image.BICUBIC)
    return img


class MyDataset(Dataset):
    '''
    制作dataset，在dataset中进行剪切，反转，resize等图像增强，并转为torch格式和归一化
    '''
    def __init__(self, image_list, root_path,
                 input_height=128, input_width=None, output_height=128, output_width=None,
                 crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False):
        super(MyDataset, self).__init__()

        self.root_path = root_path
        self.image_filenames = image_list
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.is_gray = is_gray

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.crop_height = crop_height
        self.crop_width = crop_width

        self.input_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = load_image(os.path.join(self.root_path, self.image_filenames[index]),
                         self.input_height, self.input_width, self.output_height, self.output_width,
                         self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)
        img = self.input_transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)


class MyDataLoader(object):
    '''
    制作有next功能的dataloader
    '''
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 制作调用next函数时的无限循环
        self.unlimit_gen = self.generator(True)

    def generator(self, inf=False):
        while True:
            data_loader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=self.drop_last)
            for images in data_loader:
                yield images
            if not inf:
                break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return len(self.dataset)//self.batch_size


if __name__ == '__main__':
    ms_ssim = MS_SSIM(max_val=1)
    root = 'F:/Work/Dataset/celebA/img_celeba/'
    image_list = [x for x in os.listdir(root)]
    random.shuffle(image_list)
    image_list = image_list

    dataset = MyDataset(image_list, root, input_height=None, crop_height=None, output_height=128, is_mirror=False)
    data_loader = MyDataLoader(dataset, 100, drop_last=False, shuffle=True)

    total_msssim = []
    for i in tqdm(range(10000 // 100)):
        image_1 = data_loader.next().cuda()
        image_2 = data_loader.next().cuda()
        sub_msssim = ms_ssim(image_1, image_2)
        total_msssim.append(sub_msssim.item())
    total_msssim = np.array(total_msssim)
    print(total_msssim)
    print('平均ms_ssim:', np.mean(total_msssim))

    # ms_ssim = MS_SSIM(max_val=1)
    #
    # root_path = 'F:/Work/Dataset/celebAHQ/celeba-128/'
    # rec_path = 'F:/Work/myGitHub/GAN-ZOO/IntroVAE/samples/rec/'
    # image_list = [x for x in os.listdir(rec_path)]
    #
    # target_dataset = MyDataset(image_list, root_path, input_height=None, crop_height=None, output_height=128, is_mirror=False)
    # target_dataloader = MyDataLoader(target_dataset, 1, drop_last=False, shuffle=False)
    # rec_dataset = MyDataset(image_list, rec_path, input_height=None, crop_height=None, output_height=128, is_mirror=False)
    # rec_dataloader = MyDataLoader(rec_dataset, 1, drop_last=False, shuffle=False)
    #
    # total_msssim = []
    # for i in tqdm(range(len(image_list))):
    #     target = target_dataloader.next().cuda()
    #     rec = rec_dataloader.next().cuda()
    #     sub_msssim = ms_ssim(target, rec)
    #     total_msssim.append(sub_msssim.item())
    # total_msssim = np.array(total_msssim)
    # print(total_msssim)
    # print('平均ms_ssim:', np.mean(total_msssim))

