import cv2
import torch.utils.data as data
import torch
import os
import os.path
import numpy as np
import json
import sys

class PointRGBRecord(object):
    def __init__(self, path, height):
        self.path = path
        self.episode, self.camera, self.number = self.path.split('/')
        self.height = height

    def RGB_FRONT_input(self):
        return os.path.join(self.episode, 'CAM_RGB_FRONT', self.number)

    def RGB_FRONT_RIGHT_input(self):
        return os.path.join(self.episode, 'CAM_RGB_FRONT_RIGHT', self.number)

    def RGB_FRONT_LEFT_input(self):
        return os.path.join(self.episode, 'CAM_RGB_FRONT_LEFT', self.number)

    def RGB_BACK_RIGHT_input(self):
        return os.path.join(self.episode, 'CAM_RGB_BACK_RIGHT', self.number)

    def RGB_BACK_LEFT_input(self):
        return os.path.join(self.episode, 'CAM_RGB_BACK_LEFT', self.number)

    def RGB_BACK_input(self):
        return os.path.join(self.episode, 'CAM_RGB_BACK', self.number)

    @property
    def OverviewMask(self):
        return os.path.join(self.episode, 'CAM_SemSeg_TOPDOWN_{}'.format(self.height), self.number)

    @property
    def OverviewMap(self):
        return os.path.join(self.episode, 'CAM_RGB_TOPDOWN_{}'.format(self.height), self.number)

    @property
    def video_frame(self):
        return os.path.join(self.path, 'topdown_video_frame.png')

    @property
    def get_number(self):
        return self.number.split('.')[0]


class Carla_Dataset(data.Dataset):
    def __init__(self, datadir, split, transform, is_train=True,
                 num_views=8, input_size=224, label_size=25,
                 use_mask=False, use_depth=False, height=10):
        self.datadir = datadir
        self.split = split
        self.transform = transform
        self.is_train = is_train
        self.num_views = num_views
        self.input_size = input_size
        self.label_size = label_size
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.height = height
        print('use mask: ', self.use_mask)
        print('use depth: ', self.use_depth)
        self._parse_list()
        self._parse_color()

    def _parse_list(self):
        tmp = []
        for x in open(self.split):
            x = x.strip()
            tmp.append(x)
        self.coor_list = [PointRGBRecord(item, height=self.height) for item in tmp]
        print('Coordinate number:%d' % (len(self.coor_list)))

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

        input_img = [cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_LEFT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_RIGHT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_RIGHT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_LEFT_input()))]

        # print('\n\n\n\n', os.path.join(self.datadir, example.RGB_FRONT_input()),
        #       os.path.join(self.datadir, example.RGB_FRONT_LEFT_input()),
        #                    os.path.join(self.datadir, example.RGB_FRONT_RIGHT_input()),
        #                                 os.path.join(self.datadir, example.RGB_BACK_input()),
        #                                              os.path.join(self.datadir, example.RGB_BACK_RIGHT_input()),
        #                                              os.path.join(self.datadir, example.RGB_BACK_LEFT_input()),
        #                                              '\n\n\n\n')

        split_data = []
        for img in input_img:
            resized_img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            split_data.append(resized_img)
        input_data.extend(split_data)

        image = input_data.copy()
        input_data = self.transform(input_data)

        om_orig = cv2.imread(os.path.join(self.datadir, example.OverviewMask))

        om = cv2.resize(om_orig[:, :, ::-1], (self.label_size, self.label_size), interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(om[:, :, 0])
        if not self.is_train:
            rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in image], axis=0)
            return input_data, mask.long(), rgb_origin, om_orig[:, :, -1]
        return input_data, mask.long()

    def __len__(self):
        return len(self.coor_list)


class nuScenes_Dataset(data.Dataset):
    def __init__(self, datadir, split, transform, is_train=True,
                 num_views=8, input_size=224, label_size=25,
                 use_mask=False, use_depth=False, height=10):
        self.datadir = datadir
        self.split = split
        self.transform = transform
        self.is_train = is_train
        self.num_views = num_views
        self.input_size = input_size
        self.label_size = label_size
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.height = height
        print('use mask: ', self.use_mask)
        print('use depth: ', self.use_depth)
        self._parse_list()
        self._parse_color()

    def _parse_list(self):
        tmp = []
        for x in open(self.split):
            x = x.strip()
            tmp.append(x)
        self.coor_list = [PointRGBRecord(item, height=self.height) for item in tmp]
        print('Coordinate number:%d' % (len(self.coor_list)))

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

        input_img = [cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_LEFT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_RIGHT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_RIGHT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_LEFT_input()))]

        # print('\n\n\n\n', os.path.join(self.datadir, example.RGB_FRONT_input()),
        #       os.path.join(self.datadir, example.RGB_FRONT_LEFT_input()),
        #                    os.path.join(self.datadir, example.RGB_FRONT_RIGHT_input()),
        #                                 os.path.join(self.datadir, example.RGB_BACK_input()),
        #                                              os.path.join(self.datadir, example.RGB_BACK_RIGHT_input()),
        #                                              os.path.join(self.datadir, example.RGB_BACK_LEFT_input()),
        #                                              '\n\n\n\n')
        sys.stdout.flush()

        split_data = []
        for img in input_img:
            # print(img is None)
            resized_img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            split_data.append(resized_img)
        input_data.extend(split_data)

        # image = input_data.copy()
        input_data = self.transform(input_data)

        # om_orig = cv2.imread(os.path.join(self.datadir, example.OverviewMask))
        #
        # om = cv2.resize(om_orig[:, :, ::-1], (self.label_size, self.label_size), interpolation=cv2.INTER_NEAREST)
        #
        # mask = torch.from_numpy(om[:, :, 0])
        # if not self.is_train:
        #     rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in image], axis=0)
        #     return input_data, mask.long(), rgb_origin, om_orig[:, :, -1]
        return input_data

    def __len__(self):
        return len(self.coor_list)


class Seq_OVMDataset(data.Dataset):
    def __init__(self, datadir, split, transform, is_train=True,
                 num_views=8, input_size=224, label_size=25,
                 use_mask=False, use_depth=False, height=10):
        self.datadir = datadir
        self.split = split
        self.transform = transform
        self.is_train = is_train
        self.num_views = num_views
        self.input_size = input_size
        self.label_size = label_size
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.height = height
        print('use mask: ', self.use_mask)
        print('use depth: ', self.use_depth)
        self._parse_list()
        self._parse_color()

    def _parse_list(self):
        tmp = []
        for x in open(self.split):
            x = x.strip()
            tmp.append(x)
        tmp = sorted(tmp)  # [:1000]
        self.coor_list = [PointRGBRecord(item, height=self.height) for item in tmp]
        print('Coordinate number:%d' % (len(self.coor_list)))

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
        gap = int(8 / self.num_views)
        # if self.use_mask:
        #     input_img = cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_input()))
        # elif self.use_depth:
        #     input_img = cv2.imread(os.path.join(self.datadir, example.call_input('depth', 0)))
        # else:
        input_img = [cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_RIGHT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_LEFT_input()))]
        # for i, rank in enumerate([2, 2, 3, 3, 0, 0, 1, 1]):
        #     if i % gap == 0:
        #         split_data = input_img[:, input_img.shape[1]//4*rank:input_img.shape[1]//4*(rank+1)]
        split_data = []
        for img in input_img:
            resized_img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            split_data.append(resized_img)
        input_data.extend(split_data)

        # if self.num_views > 4:
        #     if self.use_mask:
        #         input_img_45 = cv2.imread(os.path.join(self.datadir, example.call_input('sem', 45)))
        #     elif self.use_depth:
        #         input_img_45 = cv2.imread(os.path.join(self.datadir, example.call_input('depth', 45)))
        #     else:
        #         input_img_45 = cv2.imread(os.path.join(self.datadir, example.call_input('rgb', 45)))
        #     for i, rank in enumerate([2, 3, 0, 1]):
        #         split_data = input_img_45[:, input_img_45.shape[1]//4*rank:input_img_45.shape[1]//4*(rank+1)]
        #         split_data = cv2.resize(split_data, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        #         input_data[2*i+1] = split_data
        image = input_data.copy()
        input_data = self.transform(input_data)
        om_orig = cv2.imread(os.path.join(self.datadir, example.OverviewMask))
        # print('om_orig.shape', om_orig.shape)
        # print('om_orig.shape', om_orig[0])
        om = cv2.resize(om_orig[:, :, ::-1], (self.label_size, self.label_size), interpolation=cv2.INTER_NEAREST)
        # mask = np.uint8(np.zeros((self.label_size, self.label_size)))
        # for i, _ in enumerate(om):
        #     for j, _ in enumerate(om[0]):
        #         key = ','.join([str(x) for x in om[i, j]])
        #         mask[i, j] = self.label_dic[key]
        mask = torch.from_numpy(om[:, :, 0])
        if not self.is_train:
            rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in image], axis=0)
            return input_data, mask.long(), rgb_origin, om_orig[:, :, -1]
        return input_data, mask.long()

    def __len__(self):
        return len(self.coor_list)


class Inte_OVMDataset(data.Dataset):
    def __init__(self, datadir, split, transform, is_train=True,
                 num_views=8, input_size=224, label_size=25,
                 use_mask=False, use_depth=False, height=10):
        self.datadir = datadir
        self.split = split
        self.transform = transform
        self.is_train = is_train
        self.num_views = num_views
        self.input_size = input_size
        self.label_size = label_size
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.height = height
        print('use mask: ', self.use_mask)
        print('use depth: ', self.use_depth)
        self._parse_list()
        self._parse_color()

    def _parse_list(self):
        tmp = []
        for x in open(self.split):
            x = x.strip()
            tmp.append(x)
        tmp = sorted(tmp)  # [:1000]
        self.coor_list = [PointRGBRecord(item, height=self.height) for item in tmp]
        with open(os.path.join('data', x.strip().split('/')[0], 'img_info_list.json'), 'r') as f:
            self.img_info = json.load(f)  # [:100]

        print('Coordinate number:%d' % (len(self.coor_list)))

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
        input_img = [cv2.imread(os.path.join(self.datadir, example.RGB_FRONT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_RIGHT_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_BACK_input())),
                     cv2.imread(os.path.join(self.datadir, example.RGB_LEFT_input()))]
        input_img_info = self.img_info[int(example.get_number)]
        split_data = []
        for img in input_img:
            resized_img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            split_data.append(resized_img)
        input_data.extend(split_data)

        image = input_data.copy()
        input_data = self.transform(input_data)
        om_orig = cv2.imread(os.path.join(self.datadir, example.OverviewMask))
        om = cv2.resize(om_orig[:, :, ::-1], (self.label_size, self.label_size), interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(om[:, :, 0])
        if not self.is_train:
            rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in image], axis=0)
            return input_data, mask.long(), rgb_origin, om_orig[:, :, -1], input_img_info
        return input_data, mask.long(), input_img_info

    def __len__(self):
        return len(self.coor_list)