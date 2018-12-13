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
from models import VPNModel
from datasets import Seq_OVMDataset
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
import cv2
import shutil
import dominate
from dominate.tags import *
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
        num_views=args.n_views,
        num_class=94,
        transform_type=args.transform_type,
        output_size=args.label_resolution,
    )

    val_dataset = Seq_OVMDataset(args.test_dir, pix_file=args.pix_file,
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


    mapper = VPNModel(network_config)
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

    with open('./metadata/colormap_coarse.csv') as f:
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

    for step, (rgb_stack, target, rgb_origin, topmap, OverMaskOrigin) in enumerate(val_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            input_rgb_var = torch.autograd.Variable(rgb_stack).cuda()
            _, output = mapper(x=input_rgb_var, test_comb=[x * int(args.n_views / args.test_views) for x in list(range(args.test_views))], return_feat=True)
        target_var = target.cuda()
        target_var = target_var.view(-1)
        output = output.view(-1, args.num_class)
        upsample = output.view(-1, args.label_resolution, args.label_resolution, args.num_class).transpose(3,2).transpose(2,1).contiguous()
        upsample = nn.functional.upsample(upsample, size=args.segSize, mode='bilinear', align_corners=False)
        upsample = nn.functional.softmax(upsample, dim=1)
        freemap = upsample.data.index_select(dim=1, index=torch.Tensor(reachable).long().cuda())
        freemap = freemap.sum(dim=1, keepdim=False)
        output = nn.functional.log_softmax(output, dim=1)
        _, pred = upsample.data.topk(1, 1, True, True)
        pred = pred.squeeze(1)
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input_rgb_var.size(0))
        prec_stat = count_mean_accuracy(output.data, target_var.data, prec_stat)
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        top1.update(prec1[0], rgb_stack.size(0))
        top5.update(prec5[0], rgb_stack.size(0))

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

            syn_map = np.zeros((256*3, 256*4 + 40, 3))
            syn_map += 255

            pred = np.uint8(pred.cpu()[0])
            predMask = np.uint8(np.zeros((args.segSize, args.segSize, 3)))
            for i, _ in enumerate(pred):
                for j, _ in enumerate(pred[0]):
                    key = str(pred[i][j])
                    predMask[i,j] = label_dic[key]
            predMask = cv2.resize(predMask[:, :, ::-1], (256, 256), interpolation=cv2.INTER_NEAREST)

            gtMask = OverMaskOrigin[0].cpu().numpy()
            gtMask = cv2.resize(gtMask, (256, 256), interpolation=cv2.INTER_NEAREST)
            syn_map[:256, 256*3 + 40:] = gtMask
            syn_map[256:256*2, 256*3 + 40:] = predMask

            rgb = rgb_origin.cpu().numpy()[0]
            topmap = topmap.cpu().numpy()

            freemap = np.uint8(freemap[0]*255)
            freemap = cv2.applyColorMap(freemap, cv2.COLORMAP_JET)
            freemap = cv2.resize(freemap, (256, 256), interpolation=cv2.INTER_NEAREST)
            syn_map[256*2:256*3, 256*3 + 40:] = freemap

            orient_rank = [-2, 6, -2, 0, -1, 4, -2, 2, -2]
            for i, orient in enumerate(orient_rank):
                if orient >= 0:
                    syn_map[(i%3)*256:(i%3+1)*256, int(i/3)*256:int(i/3+1)*256] = cv2.resize(rgb[orient], (256, 256))
                elif orient == -1:
                    syn_map[(i%3)*256:(i%3+1)*256, int(i/3)*256:int(i/3+1)*256] = topmap[:, :, ::-1]
                elif orient == -2:
                    syn_map[(i%3)*256:(i%3+1)*256, int(i/3)*256:int(i/3+1)*256] = 240
            for i in range(2):
                syn_map[256 * (i + 1) - 3: 256 * (i + 1) + 3] = 255
            for i in range(3):
                syn_map[:, 256 * (i + 1) -3: 256 * (i + 1) + 3] = 255
            cv2.imwrite(os.path.join(web_path, 'syn_step%d.jpg'%(step+1)), syn_map)
            frames.append(syn_map[:, :, ::-1])
    clip = mpy.ImageSequenceClip(frames, fps=8)
    clip.write_videofile(os.path.join(web_path, 'OverviewSemVideo.mp4'), fps=8)


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
