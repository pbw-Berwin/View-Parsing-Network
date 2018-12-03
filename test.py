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
from datasets import OVMDataset
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
from rot_models import AVTRotModule

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

    val_dataset = OVMDataset(args.data_root, args.eval_list, pix_file=args.pix_file,
                         transform=torchvision.transforms.Compose([
                             Stack(roll=True),
                             ToTorchFormatTensor(div=True),
#                             GroupNormalize(mean_rgb, std_rgb)
                         ]),
                         n_views=network_config.num_views, resolution=args.input_resolution,
                         label_res=args.segSize, use_mask=args.use_mask, use_depth=args.use_depth, is_train=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True
    )

    # logger = Logger(os.path.join(args.tflogdir, args.logname))

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


    criterion = nn.NLLLoss(weight=None, size_average=True)
    eval(val_loader, mapper, criterion)

    web_path = os.path.join(args.visualize, args.store_name)
    if os.path.isdir(web_path):
        pass
    else:
        os.makedirs(web_path)

    with dominate.document(title=web_path) as web:
        for step in range(len(val_loader)):
            if step % args.print_freq == 0:
                h2('Step {}'.format(step*args.batch_size))
                with table(border = 1, style = 'table-layout: fixed;'):
                    with tr():
                        for i in range(args.test_views):
                            #if i % int(8 / args.n_views) == 0:
                            path = 'Step-{}-{}.png'.format(step * args.batch_size, i)
                            with td(style='word-wrap: break-word;', halign='center', valign='top'):
                                img(style='width:128px', src=path)
                        path = 'Step-{}-pred.png'.format(step * args.batch_size)
                        with td(style='word-wrap: break-word;', halign='center', valign='top'):
                            img(style='width:128px', src=path)
                        path = 'Step-{}-gt.png'.format(step * args.batch_size)
                        with td(style='word-wrap: break-word;', halign='center', valign='top'):
                            img(style='width:128px', src=path)

    with open(os.path.join(web_path, 'index.html'), 'w') as fp:
        fp.write(web.render())



def eval(val_loader, mapper, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    mapper.eval()

    end = time.time()

    web_path = os.path.join(args.visualize, args.store_name)
    if os.path.isdir(web_path):
        pass
    else:
        os.makedirs(web_path)

    prec_stat = {}
    for i in range(args.num_class):
        prec_stat[str(i)] = {'intersec': 0, 'union': 0, 'all': 0}

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

    for step, (rgb_stack, target, rgb_origin, OverMaskOrigin) in enumerate(val_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            input_rgb_var = torch.autograd.variable(rgb_stack).cuda()
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
                        fss_logit, _ = torch.stack(fss_list, dim=1).max(dim=1, keepdim=false)
                    elif args.late_fusion_type == 'avg':
                        fss_logit = torch.stack(fss_list, dim=1).mean(dim=1, keepdim=false)
                    output = torch.log(fss_logit)
                else:
                    ret_t = torch.tensor([(x * int(args.n_views / args.test_views) + args.view_bias) % 8 for x in list(range(args.test_views))])
                    input_rgb_var = torch.index_select(input_rgb_var, 1, ret_t.cuda())
                    input_rgb_var = input_rgb_var.view([input_rgb_var.size(0), input_rgb_var.size(1)*input_rgb_var.size(2)] + list(input_rgb_var.size()[3:]))
                    ret_t = ret_t.unsqueeze(0)
                    ret_t = ret_t.expand([input_rgb_var.size(0), ret_t.size(1)])
                    output = mapper(input_rgb_var, ret_t)
            else:
                _, output = mapper(x=input_rgb_var, test_comb=list(range(args.test_views)), return_feat=True)
        target_var = target.cuda()
        target_var = target_var.view(-1)
        # output = output.view(-1, args.num_class)
        upsample = output.view(-1, args.label_resolution, args.label_resolution, args.num_class).transpose(3,2).transpose(2,1).contiguous()
        upsample = nn.functional.upsample(upsample, size=args.segSize, mode='bilinear', align_corners=False)
        upsample = nn.functional.softmax(upsample, dim=1)
        output = torch.log(upsample.transpose(1,2).transpose(2,3).contiguous().view(-1, args.num_class))
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

        if step % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                step + 1, len(val_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            print(output)

            pred = np.uint8(pred.cpu()[0])
            predMask = np.uint8(np.zeros((args.segSize, args.segSize, 3)))
            for i, _ in enumerate(pred):
                for j, _ in enumerate(pred[0]):
                    key = str(pred[i][j])
                    predMask[i,j] = label_dic[key]
            predMask = cv2.resize(predMask[:, :, ::-1], (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(web_path, 'Step-{}-pred.png'.format(step * args.batch_size, i)), predMask)

            gtMask = OverMaskOrigin[0].cpu().numpy()
            gtMask = cv2.resize(gtMask, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(web_path, 'Step-{}-gt.png'.format(step * args.batch_size, i)), gtMask)

            rgb = rgb_origin.cpu().numpy()[0]
            for i in range(args.test_views):
                cv2.imwrite(os.path.join(web_path, 'Step-{}-{}.png'.format(step * args.batch_size, i)), cv2.resize(rgb[(i + args.view_bias) % 8], (256, 256), interpolation=cv2.INTER_NEAREST))

    sum_acc = 0
    counted_cat = 0
    sum_iou = 0
    for key in prec_stat:
        if int(prec_stat[key]['all']) != 0:
            acc = prec_stat[key]['intersec'] / (prec_stat[key]['all'] + 1e-10)
            iou = prec_stat[key]['intersec'] / (prec_stat[key]['union'] + 1e-10)
            sum_acc += acc
            sum_iou += iou
            counted_cat += 1
    mean_acc = sum_acc / counted_cat
    mean_iou = sum_iou / counted_cat
    output = ('Testing Results: Prec@1 {top1.avg:.3f} Mean Prec@1 {meantop:.3f} Mean IoU {meaniou:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, loss=losses, meantop=mean_acc, meaniou=mean_iou))
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
        union_t = pred_map + target_map - intersection_t
        prec_stat[key]['intersec'] += np.sum(intersection_t)
        prec_stat[key]['union'] += np.sum(union_t)
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
