model = dict(
    type='SparseInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    mask_head=dict(
        type='SparseInstHead',
        num_classes=80,
        in_channels=2048,
        img_max_shape=(416, 576),
        encoder=dict(
            type='InstanceContextEncoder',
            in_features=(1,2,3),
            input_shape=(512,1024,2048),
            num_channels=256),
        decoder=dict(
            type='BaseIAMDecoder',
            num_masks=100,
            num_classes=80,
            num_channels=256,
            kernel_dim=128,
            scale_factor=2.0,
            output_iam=False,
            dim=256,
            num_convs=4),
        criterion=dict(
            type='SparseInstCriterion',
            items=("labels", "masks"),
            num_classes = 80,
            matcher=dict(
                type='SparseInstMatcher',
                alpha=0.8,
                beta=0.2,
            ),
            loss_cfg=dict(
                class_weight=2.0,
                num_classes=80,
                mask_pixel_weight=5.0,
                mask_dice_weight=2.0,
                objectness_weight=1.0,
            )
        )
    ),
    train_cfg=dict(a=1),
    test_cfg=dict(cls_threshold=0.005,
                  mask_threshold = 0.45,
                  max_detections = 100)
)
