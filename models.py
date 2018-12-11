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
        for index in range(x.size(0))[1:]:
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


