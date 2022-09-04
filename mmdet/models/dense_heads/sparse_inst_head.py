from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList)
from ..utils import (BitMasks, masks_to_boxes, rescoring_mask, unpack_gt_instances)
from .base_mask_head import BaseMaskHead


@MODELS.register_module()
class SparseInstHead(BaseMaskHead):
    """Implements the SparseInst head.
    TODO:put the article link here:
    """

    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            encoder: ConfigType,
            decoder: ConfigType,
            criterion: ConfigType,
            train_cfg: ConfigType,
            test_cfg: ConfigType,
            img_max_shape: tuple):
        super(BaseMaskHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.img_max_shape = img_max_shape

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

    def prepare_targets(self, gt_instances):
        target = []
        for gt_instance in gt_instances:
            gt_dic = {}
            device = gt_instance.labels.device
            gt_dic['labels'] = gt_instance.labels
            gt_dic['masks'] = BitMasks(gt_instance.masks, device=device)
            target.append(gt_dic)
        return target

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False,
                results_list: OptInstanceList = None,
                **kwargs) -> InstanceList:
        """Test function without test-time augmentation.

                Args:
                    x (tuple[Tensor]): Multi-level features from the
                        upstream network, each is a 4D-tensor.
                    batch_data_samples (List[:obj:`DetDataSample`]): The Data
                        Samples. It usually includes information such as
                        `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
                    rescale (bool, optional): Whether to rescale the results.
                        Defaults to False.
                    results_list (list[obj:`InstanceData`], optional): Detection
                        results of each image after the post process. Only exist
                        if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.

                Returns:
                    list[obj:`InstanceData`]: Instance segmentation
                    results of each image after the post process.
                    Each item usually contains following keys.

                        - scores (Tensor): Classification scores, has a shape
                          (num_instance,)
                        - labels (Tensor): Has a shape (num_instances,).
                        - masks (Tensor): Processed mask results, has a
                          shape (num_instances, h, w).
                """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        results_list = self.predict_by_feat(
            **outs,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            results_list=results_list,
            **kwargs)
        return results_list

    def predict_by_feat(
            self,
            pred_logits: Tensor,
            pred_masks: Tensor,
            pred_scores: Tensor,
            batch_img_metas: [Dict],
            rescale: bool,
            results_list: List,
            **kwargs) -> InstanceList:

        if results_list == None:
            results_list = []
        pred_scores_after = pred_logits.sigmoid()
        pred_masks = pred_masks.sigmoid()
        pred_objectness = pred_scores.sigmoid()
        pred_scores = torch.sqrt(pred_scores_after * pred_objectness)
        for _, (scores_per_image, mask_pred_per_image, batched_input) \
                in enumerate(zip(pred_scores, pred_masks, batch_img_metas)):
            result = InstanceData()
            img_shape = batched_input['img_shape']
            max_shape = batched_input['img_shape']
            ori_shape = batched_input['ori_shape']
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.test_cfg.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]
            if scores.size(0) == 0:
                result.scores = scores
                result.labels = labels
                results_list.append(result)
                continue
            h, w = img_shape
            # rescoring mask using maskness
            scores = rescoring_mask(scores, mask_pred_per_image > self.test_cfg.mask_threshold, mask_pred_per_image)
            # upsample the masks to the original resolution:
            # (1) upsampling the masks to the padded inputs, remove the padding area
            # (2) upsampling/downsampling the masks to the original sizes
            mask_pred_per_image = F.interpolate(mask_pred_per_image.unsqueeze(1), size=max_shape, mode="bilinear",
                                                align_corners=False)[:, :, :h, :w]
            mask_pred_per_image = F.interpolate(mask_pred_per_image, size=ori_shape, mode='bilinear',
                                                align_corners=False).squeeze(1)
            mask_pred = mask_pred_per_image > self.test_cfg.mask_threshold
            result.masks = mask_pred
            result.scores = scores
            result.labels = labels
            result.bboxes = masks_to_boxes(mask_pred)
            results_list.append(result)
        return results_list

    # TODO：learn how to deal with some of the initial function.
    def get_targets(self, points: List[Tensor],
                    batch_gt_instances: InstanceList) -> int:
        pass

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        pass
