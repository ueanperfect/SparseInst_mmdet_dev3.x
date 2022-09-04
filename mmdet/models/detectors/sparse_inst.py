# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
from .single_stage_instance_seg import SingleStageInstanceSegmentor
import torch


@MODELS.register_module()
class SparseInst(SingleStageInstanceSegmentor):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""
    def __init__(self,
                 backbone: ConfigType,
                 mask_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,):
        super().__init__(
            backbone=backbone,
            neck=None,
            bbox_head=None,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            )
