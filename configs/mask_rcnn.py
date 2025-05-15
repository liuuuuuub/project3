_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# 修改模型配置为20类
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),

    roi_head=dict(
        bbox_head=dict(num_classes=20),  # 20 classes + background
        mask_head=dict(num_classes=20)   # 20 classes + background
    )
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
    dict(type='TensorboardVisBackend', save_dir='./outputs/mask_rcnn_coco2007/tensorboard')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),  # 统一大小
    dict(type='RandomFlip', flip_ratio=0.5),
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
# 数据集路径配置
dataset_type = 'CocoDataset'
data_root = './data/COCO/'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train2007.json',
        data_prefix=dict(img=data_root + 'train2007/'),
        classes=('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        pipeline=train_pipeline   ),  # 修改为 data_prefix
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
     

# 优化器配置
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)  # 适当调节 max_norm 值
)
# 学习率调度器配置
lr_config = dict(policy='step', step=[28, 32])
total_epochs = 36

# 日志保存路径
work_dir = './outputs/mask_rcnn_coco2007'



# 验证配置：计算 mAP 并记录
evaluation = dict(
    interval=1,  # 每个 epoch 结束后进行一次验证
    metric=['bbox', 'segm'],  # 同时计算目标检测（bbox）和实例分割（segm） mAP
    save_best='bbox_mAP'  # 保存最高 mAP 模型
)

# 自定义训练和验证流程（ValLoop）
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val2007.json',
    metric=['bbox', 'segm'],
    classwise=True,
    proposal_nums=[1, 10, 100],
)
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
)


val_dataloader = dict(
    # 显式添加dataset配置 ▼
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val2007.json',
        data_prefix=dict(img=data_root + 'val2007/'),
        test_mode=True , # 验证集必须设置test_mode=True ▼
        pipeline=test_pipeline
    ),
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)  # 验证集应关闭shuffle ▼
)


val_cfg = dict(
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

