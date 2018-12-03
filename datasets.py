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
        return os.path.join(self.path, 'Overview_mode=sem.png')

    @property
    def OverviewMap(self):
        return os.path.join(self.path, 'topdown_view.png')

    @property
    def video_frame(self):
        return os.path.join(self.path, 'topdown_video_frame.png')


class RealRGBRecord(object):
    def __init__(self, path):
        self.path = path

    def call_input(self, mode, yaw):
        # print(self.path)
        return os.path.join(self.path, '%s-%d.png'%(mode, yaw))

    @property
    def OverviewMask(self):
        return os.path.join(self.path, 'topdown-semantics.png')

    @property
    def OverviewRGB(self):
        return os.path.join(self.path, 'topdown-rgb_filled.png')

class OVMDataset(data.Dataset):
    def __init__(self, root_data, list_file, pix_file, transform,
            is_train=True, n_views=8, resolution=224, label_res=128,
            use_mask=False, use_depth=False, test_view_num=None, rotate=False):
        self.root_data = root_data
        self.list_file = list_file
        self.transform = transform
        self.is_train = is_train
        self.n_views = n_views
        self.resolution = resolution
        self.label_res = label_res
        self.pix_file = pix_file
        self.use_mask = use_mask
        self.use_depth= use_depth
        self.rotate=rotate
        if test_view_num is not None:
            self.n_views = test_view_num
        if self.use_mask:
            print('Use Mask')
        self._parse_list()


    def _parse_list(self):
        # tmp = [x.strip() for x in open(self.list_file)]
        tmp = []
        for x in open(self.list_file):
            x = x.strip()
            if x.split('/')[0] != 'e042c74158a0b1dad5f1b6a689fd056a':
                tmp.append(x)
        # tmp = [item for item in tmp if int(item[1])>=3]
        self.coor_list = [PointRGBRecord(item) for item in tmp]
        #if not self.is_train:
        if self.rotate:
            print('Coordinate number:%d'%(len(self.coor_list)*8))
        else:
            print('Coordinate number:%d'%(len(self.coor_list)))

        with open('/data/vision/oliva/scenedataset/activevision/House3D/House3D/metadata/colormap_coarse.csv') as f:
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
        rgb = list()
        gap = int(8/self.n_views)
        if self.use_mask:
            input_img = cv2.imread(os.path.join(self.root_data, example.call_input('sem', 0)))
        elif self.use_depth:
            input_img = cv2.imread(os.path.join(self.root_data, example.call_input('depth', 0)))
        else:
            input_img = cv2.imread(os.path.join(self.root_data, example.call_input('rgb', 0)))
        for i, rank in enumerate([2, 2, 3, 3, 0, 0, 1, 1]):
            if i % gap == 0:
                split = input_img[:, input_img.shape[1]//4*rank:input_img.shape[1]//4*(rank+1)]
                split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                rgb.extend([split])
        if self.n_views > 4:
            if self.use_mask:
                input_img_45 = cv2.imread(os.path.join(self.root_data, example.call_input('sem', 45)))
            elif self.use_depth:
                input_img_45 = cv2.imread(os.path.join(self.root_data, example.call_input('depth', 45)))
            else:
                input_img_45 = cv2.imread(os.path.join(self.root_data, example.call_input('rgb', 45)))
            for i, rank in enumerate([2, 3, 0, 1]):
                split = input_img_45[:, input_img_45.shape[1]//4*rank:input_img_45.shape[1]//4*(rank+1)]
                split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                rgb[2*i+1] = split
        image = rgb
        image_data = self.transform(image)
        OverviewMaskOriginal = cv2.imread(os.path.join(self.root_data, example.OverviewMask))
        # print(example.OverviewMask)
        OverviewMask = cv2.resize(OverviewMaskOriginal[:, :, ::-1], (self.label_res, self.label_res), interpolation=cv2.INTER_NEAREST)
        Mask = np.uint8(np.zeros((self.label_res, self.label_res)))
        for i, _ in enumerate(OverviewMask):
            for j, _ in enumerate(OverviewMask[0]):
                key = ','.join([str(x) for x in OverviewMask[i, j]])
                Mask[i, j] = self.label_dic[key]
        Mask = torch.from_numpy(Mask)
        if not self.is_train:
            rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in rgb], axis=0)
            return image_data, Mask.long(), rgb_origin, OverviewMaskOriginal
        return image_data, Mask.long()

    def __len__(self):
        if self.rotate:
            return len(self.coor_list) * 8
        return len(self.coor_list)


class Seq_OVMDataset(data.Dataset):
    def __init__(self, datadir, pix_file, transform,
            is_train=True, n_views=8, resolution=224, label_res=128,
            use_mask=False, test_view_num=None, rotate=None):
        self.datadir = datadir
        self.transform = transform
        self.is_train = is_train
        self.n_views = n_views
        self.resolution = resolution
        self.label_res = label_res
        self.pix_file = pix_file
        self.use_mask = use_mask
        self.rotate=rotate
        if test_view_num is not None:
            self.n_views = test_view_num
        if self.use_mask:
            print('Use Mask')
        self._parse_list()

    def _parse_list(self):
        # tmp = [x.strip() for x in open(self.list_file)]
        # tmp = [item for item in tmp if int(item[1])>=3]
        traj_len = len(os.listdir(self.datadir))
        self.coor_list = [PointRGBRecord(os.path.join(self.datadir, '{}'.format(i))) for i in range(traj_len)]
        # self.coor_list = [PointRGBRecord(item) for item in tmp]
        print('Coordinate number:%d'%(len(self.coor_list)))

        with open('/data/vision/oliva/scenedataset/activevision/House3D/House3D/metadata/colormap_coarse.csv') as f:
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
        rgb = list()
        gap = int(8/self.n_views)
        if not self.use_mask:
            input_img = cv2.imread(example.call_input('rgb', 0))
        else:
            input_img = cv2.imread(example.call_input('sem', 0))
        for i, rank in enumerate([2, 2, 3, 3, 0, 0, 1, 1]):
            if i % gap == 0:
                split = input_img[:, input_img.shape[1]//4*rank:input_img.shape[1]//4*(rank+1)]
                split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                rgb.extend([split])
        if self.n_views > 4:
            if not self.use_mask:
                input_img_45 = cv2.imread(example.call_input('rgb', 45))
            else:
                input_img_45 = cv2.imread(example.call_input('sem', 45))
            for i, rank in enumerate([2, 3, 0, 1]):
                split = input_img_45[:, input_img_45.shape[1]//4*rank:input_img_45.shape[1]//4*(rank+1)]
                split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                rgb[2*i+1] = split
        image = rgb
        image_data = self.transform(image)
        OverviewMaskOriginal = cv2.imread(example.OverviewMask)
        # print(example.OverviewMask)
        OverviewMask = cv2.resize(OverviewMaskOriginal[:, :, ::-1], (self.label_res, self.label_res), interpolation=cv2.INTER_NEAREST)
        Mask = np.uint8(np.zeros((self.label_res, self.label_res)))
        for i, _ in enumerate(OverviewMask):
            for j, _ in enumerate(OverviewMask[0]):
                key = ','.join([str(x) for x in OverviewMask[i, j]])
                Mask[i, j] = self.label_dic[key]
        Mask = torch.from_numpy(Mask)
        if not self.is_train:
            origin_list = list()
            input_rgb = cv2.imread(example.call_input('rgb', 0))
            for i, rank in enumerate([2, 2, 3, 3, 0, 0, 1, 1]):
                if i % gap == 0:
                    split = input_rgb[:, input_img.shape[1]//4*rank:input_img.shape[1]//4*(rank+1)]
                    split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    origin_list.extend([split])
            if self.n_views > 4:
                input_rgb = cv2.imread(example.call_input('rgb', 45))
                for i, rank in enumerate([2, 3, 0, 1]):
                    split = input_rgb[:, input_img_45.shape[1]//4*rank:input_img_45.shape[1]//4*(rank+1)]
                    split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    origin_list[2*i+1] = split

            rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in origin_list], axis=0)
            topmap = cv2.imread(example.video_frame)
            topmap = cv2.resize(topmap, (256, 256), interpolation=cv2.INTER_NEAREST)
            for _ in range(2):
                topmap = topmap[:, ::-1]
                topmap = topmap.transpose((1,0,2)).copy()
            topmap = torch.from_numpy(topmap[:, :, ::-1].copy()).long()
            return image_data, Mask.long(), rgb_origin, topmap, OverviewMaskOriginal
        return image_data, Mask.long()

    def __len__(self):
        return len(self.coor_list)


class Inte_OVMDataset(data.Dataset):
    def __init__(self, datadir, pix_file, transform, trajectory_file=None,
            is_train=True, n_views=8, centralSize=12, ppi=4, mapSize=1000, resolution=224, label_res=128,
            use_mask=False, test_view_num=None, rotate=None):
        self.datadir = datadir
        self.transform = transform
        self.is_train = is_train
        self.n_views = n_views
        self.resolution = resolution
        self.label_res = label_res
        self.pix_file = pix_file
        self.use_mask = use_mask
        self.rotate=rotate
        self.centralSize=centralSize
        self.ppi=ppi
        self.mapSize=mapSize
        self.trajectory_file = trajectory_file
        if test_view_num is not None:
            self.n_views = test_view_num
        if self.use_mask:
            print('Use Mask')
        self._parse_list()

    def _parse_list(self):
        # tmp = [x.strip() for x in open(self.list_file)]
        # tmp = [item for item in tmp if int(item[1])>=3]
        i = 0
        j = 0
        self.coor_list = []
        if self.trajectory_file is None:
            while (self.centralSize + i*self.ppi) < self.mapSize:
                while (self.centralSize + j*self.ppi) < self.mapSize:
                    self.coor_list.append(PointRGBRecord(os.path.join(self.datadir, \
                            'x_{}_y_{}'.format(self.centralSize // 2 + i*self.ppi, self.centralSize // 2 + j*self.ppi))))
                    j += 1
                i += 1
                j = 0
        else:
            import json
            with open(self.trajectory_file, 'r') as f:
                traj = json.load(f)
            for coor in traj:
                x, y = coor
                self.coor_list.append(PointRGBRecord(os.path.join(self.datadir, \
                        'x_{}_y_{}'.format(x, y))))
        print('Coordinate number:%d'%(len(self.coor_list)))

        with open('/data/vision/oliva/scenedataset/activevision/House3D/House3D/metadata/colormap_coarse.csv') as f:
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
        rgb = list()
        gap = int(8/self.n_views)
        if not self.use_mask:
            input_img = cv2.imread(example.call_input('rgb', 0))
        else:
            # print(example.call_input('sem', 0))
            # raise NotImplementedError
            input_img = cv2.imread(example.call_input('sem', 0))
        for i, rank in enumerate([2, 2, 3, 3, 0, 0, 1, 1]):
            if i % gap == 0:
                split = input_img[:, input_img.shape[1]//4*rank:input_img.shape[1]//4*(rank+1)]
                split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                rgb.extend([split])
        if self.n_views > 4:
            if not self.use_mask:
                input_img_45 = cv2.imread(example.call_input('rgb', 45))
            else:
                input_img_45 = cv2.imread(example.call_input('sem', 45))
            for i, rank in enumerate([2, 3, 0, 1]):
                split = input_img_45[:, input_img_45.shape[1]//4*rank:input_img_45.shape[1]//4*(rank+1)]
                split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                rgb[2*i+1] = split
        image = rgb
        image_data = self.transform(image)
        OverviewMaskOriginal = cv2.imread(example.OverviewMask)
        # print(example.OverviewMask)
        OverviewMask = cv2.resize(OverviewMaskOriginal[:, :, ::-1], (self.label_res, self.label_res), interpolation=cv2.INTER_NEAREST)
        Mask = np.uint8(np.zeros((self.label_res, self.label_res)))
        for i, _ in enumerate(OverviewMask):
            for j, _ in enumerate(OverviewMask[0]):
                key = ','.join([str(x) for x in OverviewMask[i, j]])
                Mask[i, j] = self.label_dic[key]
        Mask = torch.from_numpy(Mask)
        if not self.is_train:
            origin_list = list()
            input_rgb = cv2.imread(example.call_input('rgb', 0))
            for i, rank in enumerate([2, 2, 3, 3, 0, 0, 1, 1]):
                if i % gap == 0:
                    split = input_rgb[:, input_img.shape[1]//4*rank:input_img.shape[1]//4*(rank+1)]
                    split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    origin_list.extend([split])
            if self.n_views > 4:
                input_rgb = cv2.imread(example.call_input('rgb', 45))
                for i, rank in enumerate([2, 3, 0, 1]):
                    split = input_rgb[:, input_img_45.shape[1]//4*rank:input_img_45.shape[1]//4*(rank+1)]
                    split = cv2.resize(split, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
                    origin_list[2*i+1] = split

            rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in origin_list], axis=0)
            return image_data, Mask.long(), rgb_origin, OverviewMaskOriginal
        return image_data, Mask.long()

    def __len__(self):
        return len(self.coor_list)


class Real_OVMDataset(data.Dataset):
    def __init__(self, root_dir, color_file, merge_file, transform,
            n_views=8, resolution=224, label_res=128, segSize=256,
            use_mask=False, use_depth=False):
        self.root_dir = root_dir
        self.transform = transform
        self.n_views = n_views
        self.resolution = resolution
        self.label_res = label_res
        self.color_file = color_file
        self.merge_file = merge_file
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.segSize = segSize
        if self.use_depth:
            print('Use Depth')
        if self.use_mask:
            print('Use Mask')
        self._parse_list()


    def _parse_list(self):
        # tmp = [x.strip() for x in open(self.list_file)]
        # tmp = []
        scenes = os.listdir(self.root_dir)
        tmp = []
        for scene in scenes:
            for point in os.listdir(os.path.join(self.root_dir, scene)):
                tmp.append(os.path.join(scene, point))
        # for x in open(self.list_file):
        #    x = x.strip()
        #    if x.split('/')[0] != 'e042c74158a0b1dad5f1b6a689fd056a':
        #        tmp.append(x)
        # tmp = [item for item in tmp if int(item[1])>=3]
        self.coor_list = [RealRGBRecord(os.path.join(self.root_dir, item)) for item in tmp]
        # if not self.is_train:
        print('Coordinate number:%d'%(len(self.coor_list)))

        with open(self.merge_file) as f:
            lines = f.readlines()
        raw_cat = []
        for line in lines:
            line = line.rstrip()
            raw_cat.append(line)
        raw_cat = raw_cat[1:]
        self.merge = {}
        for i, value in enumerate(raw_cat):
            key = value.split(',')[0]
            self.merge[key] = value.split(',')[-2]

        with open(self.color_file) as f:
            lines = f.readlines()
        cat = []
        for line in lines:
            line = line.rstrip()
            cat.append(line)
        cat = cat[1:]
        self.color = {}
        self.mapping = {}
        for i, value in enumerate(cat):
            key = value.split(',')[0]
            self.color[key] = [int(x) for x in value.split(',')[-3:]]
            self.mapping[key] = value.split(',')[2]

        with open('./colormap_coarse.csv') as f:
            lines = f.readlines()
        h3d_cat = []
        for line in lines:
            line = line.rstrip()
            h3d_cat.append(line)
        h3d_cat = h3d_cat[1:]
        self.h3dmapping = {}
        for i, value in enumerate(h3d_cat):
            key = value.split(',')[0]
            self.h3dmapping[key] = str(i)

        for key in self.mapping.keys():
            self.mapping[key] = self.h3dmapping[self.mapping[key]]

    def cat_merge(self, input_img, merge):
        # raise NotImplementedError
        # input_img: r, g, b
        raw_mask = input_img[:, :, 2] + input_img[:, :, 1]*256 + input_img[:, :, 0]*256*256
        mask = np.uint8(np.zeros_like(raw_mask))
        for index in merge.keys():
            mp3d_Ind =  int(merge[index])
            index = int(index)
            mask += np.uint8(raw_mask == index) * mp3d_Ind
        return mask

    def mp3d2house3d(self, input_mask, mapping):
        trans_mask = np.uint8(np.zeros_like(input_mask))
        for index in mapping.keys():
            house_Ind = int(mapping[index])
            index = int(index)
            trans_mask += np.uint8(input_mask == index) * house_Ind
        return trans_mask

    def color_encoder(self, input_img, color=None):
        if color is None:
            color = self.color
        color_mask = np.uint8(np.zeros((input_img.shape[0], input_img.shape[1], 3)))
        for index in color.keys():
            for i in range(3):
                color_mask[:, :, i] += np.uint8(input_img == int(index)) * color[index][i]
        return color_mask

    def adaptdepth(self, input_img):
        depth = np.uint16(np.zeros((512, 512)))
        depth += np.uint16(input_img[:, :, 0]) + np.uint16(input_img[:, :, 1]) * 255 + np.uint16(input_img[:, :, 2]) * 255 * 255
        #print(np.max(depth))
        #raise NotImplementedError
        # print(0.1 * item)
        input_img = np.uint8(depth * 10.8 / 255.0)
        # print(np.max(input_img))
        input_img = np.expand_dims(input_img, axis=2)
        input_img = np.concatenate((input_img, input_img, input_img), axis=-1)
        # raise NotImplementedError
        return input_img

    def __getitem__(self, item):
        # print(0.1 * item)
        example = self.coor_list[item]
        images = list()
        gap = int(8/self.n_views)
        # if self.use_mask:
        for yaw in [gap*i*45 for i in range(self.n_views)]:
            # input_img = cv2.imread(example.call_input('semantics' if self.use_mask else 'rgb_filled', yaw))
            if self.use_mask:
                input_img = cv2.imread(example.call_input('semantics', yaw))
                input_img = self.cat_merge(input_img, self.merge)
                input_img = self.color_encoder(input_img, self.color)
            elif self.use_depth:
                input_img = cv2.imread(example.call_input('depth', yaw))
                input_img = self.adaptdepth(input_img)
            else:
                input_img = cv2.imread(example.call_input('rgb_filled', yaw))
            input_img = cv2.resize(input_img, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
            images.extend([input_img[:,:,::-1]])
        image_data = self.transform(images)
        OverviewMaskOrig = cv2.imread(example.OverviewMask)
        OverviewMask = cv2.resize(OverviewMaskOrig,
                                (self.segSize, self.segSize), interpolation=cv2.INTER_NEAREST)
        Mask = self.cat_merge(OverviewMask, self.merge)
        # h3d_Mask = self.mp3d2house3d(Mask, self.mapping)
        ColorMask = self.color_encoder(Mask, self.color)[:,:,::-1].copy()
        Mask = torch.from_numpy(Mask)
        rgb_origin = np.concatenate([np.expand_dims(x, 0) for x in images], axis=0)
        return image_data, Mask.long(), ColorMask, rgb_origin

    def __len__(self):
        return len(self.coor_list)




