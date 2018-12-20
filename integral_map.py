#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test.py
# Author : Bowen Pan
# Email  : panbowen0607@gmail.com
# Date   : 09/25/2018
#
# Distributed under terms of the MIT license.

"""

"""
from utils import Foo
from models import AVTModule
from datasets import OVMDataset, Seq_OVMDataset, Inte_OVMDataset
from opts import parser
from transform import *
import torchvision
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
import os
import time
from torch.nn.utils import clip_grad_norm
# from examples.cognitive_mapping.Logger import Logger
import cv2
import shutil
import dominate
from dominate.tags import *
#from rot_models import AVTRotModule
import moviepy.editor as mpy


mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]

def main():
    global args, web_path, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    network_config = Foo(
        encoder=args.encoder,
        decoder=args.decoder,
        fc_dim=args.fc_dim,
        weights_encoder='',
        weights_decoder='',
        num_views=args.n_views,
        train=True,
        use_mask=args.use_mask,
        num_class=94,
        transform_type=args.transform_type,
        consensus_type=args.consensus_type,
        intermedia_size=args.label_resolution,
        random_pick=False,
        center_loss=False,
        transpose_output=args.transpose_output,
    )

    val_dataset = Inte_OVMDataset(args.test_dir, pix_file=args.pix_file,
                        transform=torchvision.transforms.Compose([
                            Stack(roll=True),
                            ToTorchFormatTensor(div=True),
                            GroupNormalize(mean_rgb, std_rgb)
                            ]),
                        n_views=network_config.num_views, resolution=args.input_resolution,
                        label_res=args.label_resolution, use_mask=args.use_mask, is_train=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, pin_memory=True
    )


    if args.rotate:
        mapper = AVTRotModule(network_config)
    else:
        mapper = AVTModule(network_config)

    mapper = nn.DataParallel(mapper.cuda())
    if args.weights:
        if os.path.isfile(args.weights):
            print(("=> loading checkpoint '{}'".format(args.weights)))
            checkpoint = torch.load(args.weights)
            args.start_epoch = checkpoint['epoch']
            mapper.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.weights)))

    web_path = os.path.join(args.visualize, args.store_name)
    criterion = nn.NLLLoss(weight=None, size_average=True)
    eval(val_loader, mapper, criterion, web_path)

    web_path = os.path.join(args.visualize, args.store_name)


def eval(val_loader, mapper, criterion, web_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    mapper.eval()

    end = time.time()
    if os.path.isdir(web_path):
        pass
    else:
        os.makedirs(web_path)

    frames = []

    prec_stat = {}
    for i in range(args.num_class):
        prec_stat[str(i)] = {'intersec': 0, 'all': 0}

    with open('/data/vision/oliva/scenedataset/activevision/House3D/House3D/metadata/colormap_coarse.csv') as f:
        lines = f.readlines()
    cat = []
    for line in lines:
        line = line.rstrip()
        cat.append(line)
    cat = cat[1:]
    label_dic = {}
    for i, value in enumerate(cat):
        key = str(i)
        label_dic[key] = [int(x) for x in value.split(',')[1:]]

    reachable = [10, 40, 43, 45, 64, 80]

    grid_size = (args.mapSize - args.centralSize) // args.ppi
    recon_map = np.uint8(np.zeros((args.mapSize, args.mapSize)))
    conf_map = np.zeros((args.mapSize, args.mapSize))

    for step, (rgb_stack, target, rgb_origin, OverMaskOrigin) in enumerate(val_loader):
        coor_x = step // grid_size
        coor_y = step % grid_size
        coor_x = coor_x * args.ppi
        coor_y = coor_y * args.ppi
        data_time.update(time.time() - end)
        with torch.no_grad():
            input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()
            if args.rotate:
                input_rgb_var = input_rgb_var.view([-1, args.n_views, input_rgb_var.size(1) // args.n_views] + list(input_rgb_var.shape[2:]))
                if args.late_fusion:
                    fss_list = []
                    for view_ind in [(x * args.n_views // args.test_views + args.view_bias) % 8 for x in list(range(args.test_views))]:
                        ret_t = torch.tensor([view_ind])
                        input_rgb_sp = torch.index_select(input_rgb_var, 1, ret_t.cuda())
                        input_rgb_sp = input_rgb_sp.view([input_rgb_sp.size(0), input_rgb_sp.size(1)*input_rgb_sp.size(2)] + list(input_rgb_sp.size()[3:]))
                        ret_t = ret_t.unsqueeze(0)
                        ret_t = ret_t.expand([input_rgb_sp.size(0), ret_t.size(1)])
                        output = mapper(input_rgb_sp, ret_t)
                        output = torch.exp(output)
                        fss_list.append(output)
                    if args.late_fusion_type == 'max':
                        fss_logit, _ = torch.stack(fss_list, dim=1).max(dim=1, keepdim=False)
                    elif args.late_fusion_type == 'avg':
                        fss_logit = torch.stack(fss_list, dim=1).mean(dim=1, keepdim=False)
                    output = torch.log(fss_logit)
                else:
                    ret_t = torch.tensor([(x * int(args.n_views / args.test_views) + args.view_bias) % 8 for x in list(range(args.test_views))])
                    input_rgb_var = torch.index_select(input_rgb_var, 1, ret_t.cuda())
                    input_rgb_var = input_rgb_var.view([input_rgb_var.size(0), input_rgb_var.size(1)*input_rgb_var.size(2)] + list(input_rgb_var.size()[3:]))
                    ret_t = ret_t.unsqueeze(0)
                    ret_t = ret_t.expand([input_rgb_var.size(0), ret_t.size(1)])
                    output = mapper(input_rgb_var, ret_t)
            else:
                _, output = mapper(input_rgb_var, test_comb=[x * int(args.n_views / args.test_views) for x in list(range(args.test_views))], return_feat=True)
        target_var = target.cuda()
        target_var = target_var.view(-1)
        output = output.view(-1, args.num_class)
        upsample = output.view(-1, args.label_resolution, args.label_resolution, args.num_class).transpose(3,2).transpose(2,1).contiguous()
        upsample = nn.functional.upsample(upsample, size=args.segSize, mode='bilinear', align_corners=False)
        upsample = nn.functional.softmax(upsample, dim=1)
        output = nn.functional.log_softmax(output, dim=1)
        conf, pred = upsample.data.topk(1, 1, True, True)
        # print(pred.size())
        # raise NotImplementedError
        pred = pred.squeeze(1)
        conf = conf.squeeze(1)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input_rgb_var.size(0))
        prec_stat = count_mean_accuracy(output.data, target_var.data, prec_stat)
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        top1.update(prec1[0], rgb_stack.size(0))
        top5.update(prec5[0], rgb_stack.size(0))

        start_crop = args.segSize // 2 - args.centralSize * args.scale // 2
        end_crop = args.segSize // 2 + args.centralSize * args.scale // 2
        pred_patch = np.uint8(pred.squeeze(0).cpu())
        pred_patch = pred_patch[start_crop : end_crop, start_crop : end_crop]
        conf_patch = conf.squeeze(0).cpu().numpy()
        conf_patch = conf_patch[start_crop : end_crop, start_crop : end_crop]

        pred_patch = cv2.resize(pred_patch, (args.centralSize, args.centralSize), \
                interpolation=cv2.INTER_NEAREST)
        conf_patch = cv2.resize(conf_patch, (args.centralSize, args.centralSize), \
                interpolation=cv2.INTER_NEAREST)
        for _ in range(2):
            pred_patch = pred_patch[:, ::-1]
            pred_patch = pred_patch.transpose((1, 0)).copy()
            conf_patch = conf_patch[:, ::-1]
            conf_patch = conf_patch.transpose((1, 0)).copy()

        for i in range(pred_patch.shape[0]):
            for j in range(pred_patch.shape[1]):
                if conf_patch[i, j] > conf_map[coor_x + i, coor_y + j]:
                    conf_map[coor_x + i, coor_y + j] = conf_patch[i, j]
                    recon_map[coor_x + i, coor_y + j] = pred_patch[i, j]

        batch_time.update(time.time() - end)
        end = time.time()

        if step % 1 == 0:
            output = ('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                step + 1, len(val_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            print(output)

        if step % 1000 == 0:
            print('Generating the ckpt-step-{}'.format(step))
            mapMask = np.uint8(np.zeros((args.mapSize, args.mapSize, 3)))
            for i, _ in enumerate(recon_map):
                for j, _ in enumerate(recon_map[0]):
                    key = str(recon_map[i][j])
                    mapMask[i, j] = label_dic[key]
            mapMask = mapMask[:, :, ::-1]
            cv2.imwrite(os.path.join(web_path, 'predMask_whole_{}.png'.format(step)), mapMask)

            conf_heatmap = np.uint8(conf_map * 255)
            conf_heatmap = cv2.applyColorMap(conf_heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(web_path, 'pred_fss_{}.png'.format(step)), conf_heatmap)

    mapMask = np.uint8(np.zeros((args.mapSize, args.mapSize, 3)))
    for i, _ in enumerate(recon_map):
        for j, _ in enumerate(recon_map[0]):
            key = str(recon_map[i][j])
            mapMask[i, j] = label_dic[key]
    mapMask = mapMask[:, :, ::-1]
    cv2.imwrite(os.path.join(web_path, 'predMask_whole.png'), mapMask)

    conf_map = np.uint8(conf_map * 255)
    conf_map = cv2.applyColorMap(conf_map, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(web_path, 'pred_fss.png'), conf_map)
    raise NotImplementedError
    sum_acc = 0
    counted_cat = 0
    for key in prec_stat:
        if int(prec_stat[key]['all']) != 0:
            acc = prec_stat[key]['intersec'] / (prec_stat[key]['all'] + 1e-10)
            sum_acc += acc
            counted_cat += 1
    mean_acc = sum_acc / counted_cat
    output = ('Testing Results: Prec@1 {top1.avg:.3f} Mean Prec@1 {meantop:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, loss=losses, meantop=mean_acc))
    print(output)
    output_best = '\nBest Prec@1 of: %.3f' % (best_prec1)
    print(output_best)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_mean_accuracy(output, target, prec_stat):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)
    for key in prec_stat.keys():
        label = int(key)
        pred_map = np.uint8(pred.cpu().numpy() == label)
        target_map = np.uint8(target.cpu().numpy() == label)
        intersection_t = pred_map * (pred_map == target_map)
        prec_stat[key]['intersec'] += np.sum(intersection_t)
        prec_stat[key]['all'] += np.sum(target_map)
    return prec_stat

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=='__main__':
    main()
