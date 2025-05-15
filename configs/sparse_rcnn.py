_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = './data/COCO/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # 加载掩码
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 保持比例
    dict(type='RandomFlip',prob=0.5,),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train2007.json',
        data_prefix=dict(img=data_root + 'train2007/'),
        test_mode=True,  # 验证集必须设置test_mode=True ▼
        pipeline=train_pipeline)
)

val_dataloader = dict(
    # 显式添加dataset配置 ▼
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val2007.json',
        data_prefix=dict(img=data_root + 'val2007/'),
        test_mode=True,  # 验证集必须设置test_mode=True ▼
        pipeline=test_pipeline
    ),
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)  # 验证集应关闭shuffle ▼
)


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)


vis_backends = [
    dict(type='TensorboardVisBackend', save_dir='./outputs/sparse_rcnn_coco2007-seg/tensorboard')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train2007.json',
        data_prefix=dict(img=data_root + 'train2007/'),
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        pipeline=train_pipeline),  # 修改为 data_prefix
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val2007.json',
        data_prefix=dict(img=data_root + 'val2007/'),
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val2007.json',
        data_prefix=dict(img=data_root + 'val2007/'),
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        pipeline=test_pipeline))




num_stages = 6
num_proposals = 100


model = dict(
    type='QueryInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=20,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ],
        mask_head=[
            dict(
                type='DynamicMaskHead',
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=14,
                    with_proj=False,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                num_convs=4,
                num_classes=20,
                roi_feat_size=14,
                in_channels=256,
                conv_kernel_size=3,
                conv_out_channels=256,
                class_agnostic=False,
                norm_cfg=dict(type='BN'),
                upsample_cfg=dict(type='deconv', scale_factor=2),
                loss_mask=dict(
                    type='DiceLoss',
                    loss_weight=8.0,
                    use_sigmoid=True,
                    activate=False,
                    eps=1e-5)) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1,
                mask_size=28,
            ) for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=None, rcnn=dict(max_per_img=num_proposals, mask_thr_binary=0.5)))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=0.1, norm_type=2))
work_dir = './outputs/sparse_rcnn_coco2007-seg'
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[12, 24],
        gamma=0.1)
]


total_epochs = 36
evaluation = dict(
    interval=1,  # 每个 epoch 结束后进行一次验证
    metric=['bbox','segm'],  # 同时计算目标检测（bbox）和实例分割（segm） mAP
    save_best='segm_mAP'  # 保存最高 mAP 模型
)
# 自定义训练和验证流程（ValLoop）
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val2007.json',
    metric=['bbox','segm'],
    classwise=True,
    proposal_nums=[1, 10, 100],
)
al_cfg = dict(
    type='ValLoop',
    dataloader=val_dataloader,
    evaluator=val_evaluator
    
)
# 保存最优模型
checkpoint_config = dict(
    interval=1,
    save_optimizer=True,
    max_keep_ckpts=5
)

