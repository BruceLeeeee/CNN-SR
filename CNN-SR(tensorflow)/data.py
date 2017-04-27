from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, glob, re
import scipy.io
from random import shuffle

class DataSet(object):
    def __init__(self, images_list_path, num_epoch, batch_size):
        self.image_list= self.__get_image_list(images_list_path)
        self.total_sample = len(self.image_list)

    def __get_image_list(self, data_path):
        l = glob.glob(os.path.join(data_path, "*"))
        print(len(l))
        l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
        print(len(l))
        image_list = []
        for f in l:
            if os.path.exists(f):
                if os.path.exists(f[:-4] + "_2.mat"): image_list.append([f, f[:-4] + "_2.mat"])
                if os.path.exists(f[:-4] + "_3.mat"): image_list.append([f, f[:-4] + "_3.mat"])
                if os.path.exists(f[:-4] + "_4.mat"): image_list.append([f, f[:-4] + "_4.mat"])
        return image_list

    def get_image_batch(self, offset, batch_size, img_size_0, img_size_1):
        target_list = self.image_list[offset:offset + batch_size]
        lr_list = []
        hr_list = []
        for pair in target_list:
            lr_img = scipy.io.loadmat(pair[1])['patch']
            hr_img = scipy.io.loadmat(pair[0])['patch']
            lr_list.append(lr_img)
            hr_list.append(hr_img)
        lr_list = np.array(lr_list)
        lr_list.resize([batch_size, img_size_1, img_size_0, 1])
        hr_list = np.array(hr_list)
        hr_list.resize([batch_size, img_size_1, img_size_0, 1])
        return lr_list, hr_list
