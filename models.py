#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : models.py
# Author : Bowen Pan
# Email  : panbowen0607@gmail.com
# Date   : 09/18/2018
#
# Distributed under terms of the MIT license.

import torch
from torch import nn
import utils
from collections import OrderedDict
import torch.nn.functional as F
from segmentationToolbox.models.models import *
import numpy as np
from itertools import combinations
import random

builder = ModelBuilder()

class TransformModule(nn.Module):
    def __init__(self, input_dim=25, num_view=8, consensus_type='sum', center_loss=False):
        super(TransformModule, self).__init__()
        self.num_view = num_view
        self.input_dim = input_dim
        self.consensus_type = consensus_type
        self.center_loss = center_loss
        self.mat_list = nn.ModuleList()
        for i in range(self.num_view):
            fc_transform = nn.Sequential(
                        nn.Linear(input_dim * input_dim, input_dim * input_dim),
                        nn.ReLU(),
                        nn.Linear(input_dim * input_dim, input_dim * input_dim),
                        nn.ReLU()
                    )
            self.mat_list += [fc_transform]

    def forward(self, x, test_comb=None):
        # shape x: B, V, C, H, W
        x = x.view(list(x.size()[:3]) + [self.input_dim * self.input_dim,])
        if test_comb is None:
            view_compute = np.random.randint(1, self.num_view + 1)
            comb = random.sample(list(range(self.num_view)), view_compute)
            view_comb = self.mat_list[comb[0]](x[:, comb[0]])
            t_list = [view_comb]
            for index in comb[1:]:
                t = self.mat_list[index](x[:, index])
                t_list.append(t)
                view_comb += t
                if self.consensus_type == 'avg':
                    view_comb /= len(comb)
            view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.input_dim, self.input_dim])
            if self.center_loss:
                if self.consensus_type == 'avg':
                    return view_comb, view_comb, t_list
                elif self.consensus_type == 'sum':
                    return view_comb, view_comb / len(comb), t_list
            return view_comb
        else:
            view_comb = self.mat_list[test_comb[0]](x[:, test_comb[0]])
            for index in range(len(test_comb))[1:]:
                view_comb += self.mat_list[index](x[:, index])
                if self.consensus_type == 'avg':
                    view_comb /= len(test_comb)
            view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.input_dim, self.input_dim])
            return view_comb


class SumModule(nn.Module):
    def __init__(self, input_dim=25, consensus_type='sum'):
        super(SumModule, self).__init__()
        self.input_dim = input_dim
        self.consensus_type = consensus_type

    def forward(self, x, test_comb=None):
        # shape x: B, V, C, H, W
        x = x.view(list(x.size()[:3]) + [self.input_dim * self.input_dim,])
        if test_comb is None:
            pass
        else:
            test_comb = torch.Tensor(test_comb).long().cuda()
            x = torch.index_select(x, index=test_comb, dim=1)
            x = torch.sum(x, dim=1, keepdim=False)
            #view_comb = x[:, test_comb[0]]
            #for index in range(len(test_comb))[1:]:
            #    view_comb += x[:, index]
            if self.consensus_type == 'avg':
                x /= test_comb.size(0)
            x = x.view(list(x.size()[:2]) + [self.input_dim, self.input_dim])
            return x


class AVTModule(nn.Module):
    def __init__(self, args):
        super(AVTModule, self).__init__()
        self.args = args
        self.num_views = args.num_views
        self.intermedia_size = args.intermedia_size
        self.random_pick = args.random_pick
        self.center_loss = args.center_loss
        self.consensus_type = args.consensus_type
        self.transform_type = args.transform_type
        self.transpose_output = args.transpose_output
        print('Views number: ' + str(self.num_views))
        print('Random Pick: ', self.random_pick)
        print('Center Loss: ', self.center_loss)
        print('Transform Type: ', self.transform_type)
        self.encoder = builder.build_encoder(
                    arch=args.encoder,
                    fc_dim=args.fc_dim,
                    weights=args.weights_encoder
                )
        if self.transform_type == 'fc':
            self.transform_module = TransformModule(input_dim=self.intermedia_size, consensus_type=args.consensus_type, num_view=self.num_views, center_loss=self.center_loss)
        elif self.transform_type == 'fc3':
            self.transform_module = TransformFc3Module(input_dim=self.intermedia_size, consensus_type=args.consensus_type, num_view=self.num_views, center_loss=self.center_loss)
        elif self.transform_type == 'Nonlocal':
            self.transform_module = TransformModuleNonlocal(in_planes=args.fc_dim, consensus_type=args.consensus_type, num_view=self.num_views, center_loss=self.center_loss)
        elif self.transform_type == 'Nonlocalfc':
            self.transform_module = NonlocalFCModule(in_planes=args.fc_dim, input_dim=self.intermedia_size, consensus_type=args.consensus_type, num_view=self.num_views, center_loss=self.center_loss)
        elif self.transform_type == 'sum':
            self.transform_module = SumModule(input_dim=self.intermedia_size, consensus_type=args.consensus_type)
        self.decoder = builder.build_decoder(
                    arch=args.decoder,
                    fc_dim=args.fc_dim,
                    num_class=args.num_class,
                    use_softmax=False,
                    weights=args.weights_decoder
                )

    def forward(self, x, test_comb=None, return_feat=False):
        fix_comb = None
        if not self.random_pick and test_comb is None:
            fix_comb = list(range(self.num_views))
        if test_comb is None and fix_comb is None:
            B, N, C, H, W = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] \
                                + list(x.size()[2:])).size()
        else:
            x = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] \
                                + list(x.size()[2:]))
            x = x[:, test_comb] if fix_comb is None else x[:, fix_comb]
            B, N, C, H, W = x.size()
        x = x.view(B*N, C, H, W)
        x = self.encoder(x)[0]
        x = x.view([B, N] + list(x.size()[1:]))
        if test_comb is not None:
            x = self.transform_module(x, test_comb)
        else:
            if self.center_loss:
                x, anchor, t_list = self.transform_module(x)
                anchor = anchor.view(list(anchor.size()[:2]) + [anchor.size(2) * anchor.size(3)])
            else:
                x = self.transform_module(x) if fix_comb is None else self.transform_module(x, fix_comb)
        if return_feat:
            x, feat = self.decoder([x], return_feat=return_feat)
        else:
            x = self.decoder([x])
        if self.transpose_output:
            x = x.transpose(1,2).transpose(2,3).contiguous()
            if return_feat:
                feat = feat.transpose(1,2).transpose(2,3).contiguous()
        if self.center_loss and test_comb is None:
            return x, t_list, anchor
        if return_feat:
            return x, feat
        return x


