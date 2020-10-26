# -*- coding: utf-8 -*-
# @Time    : 2020/10/26 18:28
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : ms_ssim_tf.py
# @Software: PyCharm


import os
import random
from tqdm import tqdm

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 100

def MyDataLoader(all_image_paths):
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths))
    ds = ds.shuffle(len(all_image_paths))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)

    ds = image_label_ds.cache()  # 缓存
    ds = ds.batch(batch_size)
    ds = ds.repeat(1)
    return iter(ds)


def load_and_preprocess_from_path_label(path, label=None):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)
    image /= 255.0  # 归一化到[0,1]范围
    return image


if __name__ == '__main__':
    root = 'F:/Work/myGitHub/GAN-ZOO/IntroVAE/samples/fake/'
    image_list = [os.path.join(root, x) for x in os.listdir(root)]
    random.shuffle(image_list)
    image_list = image_list

    data_loader = MyDataLoader(image_list)

    ms_ssim = []
    for i in tqdm(range(10000 // batch_size)):
        image_1 = next(data_loader)
        image_2 = next(data_loader)
        sub_ms_ssim = tf.image.ssim_multiscale(image_1, image_2, max_val=1., filter_size=7)
        ms_ssim.append(sub_ms_ssim)
    print(ms_ssim)
    print('平均ms_ssim:', tf.reduce_mean(ms_ssim))
