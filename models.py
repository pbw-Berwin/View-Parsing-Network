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
from segmentTool.models import *
import numpy as np
from itertools import combinations

builder = ModelBuilder()

class TransformModule(nn.Module):
    def __init__(self, dim=25, num_view=8):
        super(TransformModule, self).__init__()
        self.num_view = num_view
        self.dim = dim
        self.mat_list = nn.ModuleList()
        for i in range(self.num_view):
            fc_transform = nn.Sequential(
                        nn.Linear(dim * dim, dim * dim),
                        nn.ReLU(),
                        nn.Linear(dim * dim, dim * dim),
                        nn.ReLU()
                    )
            self.mat_list += [fc_transform]

    def forward(self, x):
        # shape x: B, V, C, H, W
        x = x.view(list(x.size()[:3]) + [self.dim * self.dim,])
        view_comb = self.mat_list[0](x[:, 0])
        for index in range(x.size(1))[1:]:
            view_comb += self.mat_list[index](x[:, index])
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb


class SumModule(nn.Module):
    def __init__(self):
        super(SumModule, self).__init__()

    def forward(self, x):
        # shape x: B, V, C, H, W
        x = torch.sum(x, dim=1, keepdim=False)
        return x


class VPNModel(nn.Module):
    def __init__(self, config):
        super(VPNModel, self).__init__()
        self.num_views = config.num_views
        self.output_size = config.output_size
        self.transform_type = config.transform_type
        print('Views number: ' + str(self.num_views))
        print('Transform Type: ', self.transform_type)
        self.encoder = builder.build_encoder(
                    arch=config.encoder,
                    fc_dim=config.fc_dim,
                )
        if self.transform_type == 'fc':
            self.transform_module = TransformModule(dim=self.output_size, num_view=self.num_views)
        elif self.transform_type == 'sum':
            self.transform_module = SumModule()
        self.decoder = builder.build_decoder(
                    arch=config.decoder,
                    fc_dim=config.fc_dim,
                    num_class=config.num_class,
                    use_softmax=False,
                )

    def forward(self, x, return_feat=False):
        B, N, C, H, W = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] \
                            + list(x.size()[2:])).size()

        x = x.view(B*N, C, H, W)
        x = self.encoder(x)[0]
        x = x.view([B, N] + list(x.size()[1:]))
        x = self.transform_module(x)
        if return_feat:
            x, feat = self.decoder([x], return_feat=return_feat)
        else:
            x = self.decoder([x])
        x = x.transpose(1,2).transpose(2,3).contiguous()
        if return_feat:
            feat = feat.transpose(1,2).transpose(2,3).contiguous()
            return x, feat
        return x



class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x)
        return x



