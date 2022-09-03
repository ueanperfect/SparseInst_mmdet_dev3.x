# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmengine.structures import InstanceData
from torch import Tensor
import math
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig, reduce_mean)
from ..utils import (multi_apply,c2_msra_fill,c2_xavier_fill,BitMasks)
from .anchor_free_head import AnchorFreeHead
from ..utils import (unpack_gt_instances)
from .base_mask_head import BaseMaskHead


@MODELS.register_module()
class SparseInstHead(BaseMaskHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            encoder    : ConfigType,
            decoder    : ConfigType,
            criterion  : ConfigType,
            train_cfg  : ConfigType,
            test_cfg   : ConfigType,
            img_max_shape  : tuple):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(BaseMaskHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg   = train_cfg
        self.test_cfg    = test_cfg
        self.encoder     = MODELS.build(encoder)
        self.decoder     = MODELS.build(decoder)
        self.criterion   = MODELS.build(criterion)
        self.img_max_shape   = img_max_shape

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        features = self.encoder(x)
        output = self.decoder(features)
        return output

    def loss(self,
             x: Union[List[Tensor], Tuple[Tensor]],
             batch_data_samples: SampleList,
             positive_infos: OptInstanceList = None,
             **kwargs) -> dict:
        gt_instances = unpack_gt_instances(batch_data_samples)
        output = self(x)
        targets = self.prepare_targets(gt_instances[0])
        losses = self.criterion(output, targets, self.img_max_shape)
        return losses

    def prepare_targets(self,gt_instances):
        target = []
        for gt_instance in gt_instances:
            gt_dic = {}
            device = gt_instance.labels.device
            gt_dic['labels'] = gt_instance.labels
            gt_dic['masks']  = BitMasks(gt_instance.masks,device=device)
            target.append(gt_dic)
        return target

    # def predict(self,
    #             x: Tuple[Tensor],
    #             batch_data_samples: SampleList,
    #             rescale: bool = False,
    #             results_list: OptInstanceList = None,
    #             **kwargs) -> InstanceList:
    #
    #     pass
    '''
    名填写新的bitmask
    '''
    def get_targets(self, points: List[Tensor],
                    batch_gt_instances: InstanceList) -> int:
        return 0

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        pass

    def predict_by_feat(self, *args, **kwargs):
        pass









