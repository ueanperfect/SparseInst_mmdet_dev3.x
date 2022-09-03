# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
from ...utils import c2_xavier_fill,c2_msra_fill,BitMasks
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from torch import Tensor
import math
from mmdet.registry import MODELS


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out

@MODELS.register_module()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """
    def __init__(self,
                 num_channels,
                 in_features,
                 input_shape):
        super().__init__()
        self.num_channels = num_channels
        self.in_features = in_features
        '''
        此处注意后期改写
        '''
        self.in_channels = [channel for channel in input_shape]
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        c2_msra_fill(self.fusion)

    def forward(self, features):
        features      = [feature for feature in features]
        features      = features[::-1]
        prev_features = self.ppm(self.fpn_laterals[0](features[0]))
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [outputs[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in outputs[1:]]
        features = self.fusion(torch.cat(features, dim=1))
        return features