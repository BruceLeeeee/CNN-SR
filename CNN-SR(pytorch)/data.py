import os
import glob
import re
import scipy.io
import numpy as np

import torch
import torch.utils.data as data


def get_image_list(data_path):
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


class DataSet(data.Dataset):
    def __init__(self, data_path):
        self.image_list = get_image_list(data_path)

    def __getitem__(self, index):
        item = self.image_list[index]
        lr_img = scipy.io.loadmat(item[1])['patch']
        hr_img = scipy.io.loadmat(item[0])['patch']
        lr_img = np.array(lr_img)
        hr_img = np.array(hr_img)
        lr_img.resize([1, 41, 41])
        hr_img.resize([1, 41, 41])

        return torch.FloatTensor(lr_img), torch.FloatTensor(hr_img)

    def __len__(self):
        return len(self.image_list)

