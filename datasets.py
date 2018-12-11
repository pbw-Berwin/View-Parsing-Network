import torch.utils.data as data
import torch
import os
import os.path
import numpy as np
from numpy.random import randint
import concurrent.futures
import cv2
from PIL import Image

class PointRGBRecord(object):
    def __init__(self, path):
        self.path = path

    def call_input(self, mode, yaw):
        return os.path.join(self.path, 'mode=%s_%d.png'%(mode, yaw))

    @property
    def OverviewMask(self):
        return os.path.join(self.path, 'OverviewMask.png')

    @property
    def OverviewMap(self):
        return os.path.join(self.path, 'topdown_view.png')

    @property
    def video_frame(self):
        return os.path.join(self.path, 'topdown_video_frame.png')



class OVMDataset(data.Dataset):
    def __init__(self, datadir, split, transform, is_train=True,
            num_views=8, input_size=224, label_size=25,
            use_mask=False, use_depth=False):
        self.datadir = datadir
        self.split = split
        self.transform = transform
        self.is_train = is_train
        self.num_views = num_views
        self.input_size = input_size
        self.label_size = label_size
        self.use_mask = use_mask
        self.use_depth= use_depth
        print('use mask: ', self.use_mask)
        print('use depth: ', self.use_depth)
        self._parse_list()
        self._parse_color()

    def _parse_list(self):
        tmp = []
        for x in open(self.split):
            x = x.strip()
            if x.split('/')[0] != 'e042c74158a0b1dad5f1b6a689fd056a':
                tmp.append(x)
        self.coor_list = [PointRGBRecord(item) for item in tmp]
        print('Coordinate number:%d'%(len(self.coor_list)))

    def _parse_color(self):
        with open('./metadata/colormap_coarse.csv') as f:
            lines = f.readlines()
        cat = []
        for line in lines:
            line = line.rstrip()
            cat.append(line)
        cat = cat[1:]
        self.label_dic = {}
        for i, value in enumerate(cat):
            key = ','.join(value.split(',')[1:])
            self.label_dic[key] = i

    def __getitem__(self, item):
        example = self.coor_list[item]
        input_data = list()
        gap = int(8/self.num_views)
        if self.use_mask:
            input_img = cv2.imread(os.path.join(self.datadir, example.call_input('sem', 0)))
        elif self.use_depth:
            input_img = cv2.imread(os.path.join(self.datadir, example.call_input('depth', 0)))
        else:
            input_img = cv2.imread(os.path.join(self.datadir, example.call_input('rgb', 0)))
        for i, rank in enumerate([2, 2, 3, 3, 0, 0, 1, 1]):
            if i % gap == 0:
                split_data = input_img[:, input_img.shape[1]//4*rank:input_img.shape[1]//4*(rank+1)]
                split_data = cv2.resize(split_data, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
                input_data.extend([split_data])
        if self.num_views > 4:
            if self.use_mask:
                input_img_45 = cv2.imread(os.path.join(self.datadir, example.call_input('sem', 45)))
            elif self.use_depth:
                input_img_45 = cv2.imread(os.path.join(self.datadir, example.call_input('depth', 45)))
            else:
                input_img_45 = cv2.imread(os.path.join(self.datadir, example.call_input('rgb', 45)))
            for i, rank in enumerate([2, 3, 0, 1]):
                split_data = input_img_45[:, input_img_45.shape[1]//4*rank:input_img_45.shape[1]//4*(rank+1)]
                split_data = cv2.resize(split_data, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
                input_data[2*i+1] = split_data
        image = input_data.copy()
        input_data = self.transform(input_data)
        om_orig = cv2.imread(os.path.join(self.datadir, example.OverviewMask))
        om = cv2.resize(om_orig[:, :, ::-1], (self.label_size, self.label_size), interpolation=cv2.INTER_NEAREST)
        mask = np.uint8(np.zeros((self.label_size, self.label_size)))
        for i, _ in enumerate(om):
            for j, _ in enumerate(om[0]):
                key = ','.join([str(x) for x in om[i, j]])
                mask[i, j] = self.label_dic[key]
        mask = torch.from_numpy(mask)
        if not self.is_train:
            rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in image], axis=0)
            return input_data, mask.long(), rgb_origin, om_orig
        return input_data, mask.long()

    def __len__(self):
        return len(self.coor_list)

