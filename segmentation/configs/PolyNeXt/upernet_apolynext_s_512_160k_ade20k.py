# ---------------------------------------------------------------
# UperNet + APolyNeXt-S on ADE20K
# 512x512, 160K iterations, AdamW, poly LR
# Matching the MetaFormer (ConvFormer/CAFormer) evaluation setup.
#
# Self-contained config (no _base_ inheritance).
# ---------------------------------------------------------------

norm_cfg = dict(type='SyncBN', requires_grad=True)

# APolyNeXt-S: init_channels=72, channel multipliers = {1x, 2x, 4x, 6x}
# => stage channels = [72, 144, 288, 432]

model = dict(
    type='EncoderDecoder',
    pretrained=None,  # backbone handles its own pretrained loading
    backbone=dict(
        type='PolyNeXt',
        init_channels=72,
        config_name='APolyNeXt_S',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.3,
        norm_eval=False,
        # ImageNet-1K pretrained APolyNeXt-S classification checkpoint.
        # Download from https://huggingface.co/JJWCactus/PolyNeXt
        pretrained='path/to/apolynext_s.pt',
        grad_clip_norm=5.0,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[72, 144, 288, 432],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=288,     # stage 2 output (index 2)
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)


# ---------------------------------------------------------------
# Dataset: ADE20K
# ---------------------------------------------------------------
dataset_type = 'ADE20KDataset'
data_root = 'path/to/ADEChallengeData2016'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],  # enable for multi-scale test
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
)


# ---------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'extra_norms': dict(decay_mult=0.0),
            'skip_weight': dict(decay_mult=0.0),
            'scale': dict(decay_mult=0.0),       # ScalePerChannel
            'head': dict(lr_mult=10.0),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))


# ---------------------------------------------------------------
# LR schedule: poly, 160K iterations, linear warmup
# ---------------------------------------------------------------
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

runner = dict(type='IterBasedRunner', max_iters=160000)


# ---------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------
checkpoint_config = dict(by_epoch=False, interval=16000)

evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
