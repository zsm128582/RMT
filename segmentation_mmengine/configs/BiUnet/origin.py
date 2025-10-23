checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=1)
crop_size = (
    512,
    512,
)
cudnn_benchmark = True
data = dict(
    samples_per_gpu=2,
    test=dict(
        ann_dir='annotations/validation',
        data_root='/home/zengshimao/code/RMT/data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                scale=(
                    2048,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='ADE20KDataset'),
    train=dict(
        ann_dir='annotations/training',
        data_root='/home/zengshimao/code/RMT/data/ade/ADEChallengeData2016',
        img_dir='images/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(pad_val=0, size=(
                512,
                512,
            ), type='Pad'),
            dict(type='DefaultFormatBundle'),
            dict(keys=[
                'img',
                'gt_semantic_seg',
            ], type='Collect'),
        ],
        type='ADE20KDataset'),
    val=dict(
        ann_dir='annotations/validation',
        data_root='/home/zengshimao/code/RMT/data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                flip=False,
                scale=(
                    2048,
                    512,
                ),
                transforms=[
                    dict(keep_ratio=True, type='Resize'),
                    dict(type='RandomFlip'),
                    dict(
                        mean=[
                            123.675,
                            116.28,
                            103.53,
                        ],
                        std=[
                            58.395,
                            57.12,
                            57.375,
                        ],
                        to_rgb=True,
                        type='Normalize'),
                    dict(keys=[
                        'img',
                    ], type='ImageToTensor'),
                    dict(keys=[
                        'img',
                    ], type='Collect'),
                ],
                type='MultiScaleFlipAug'),
        ],
        type='ADE20KDataset'),
    workers_per_gpu=2)
data_root = '/home/zengshimao/code/RMT/data/ade/ADEChallengeData2016'
dataset_type = 'ADE20KDataset'
default_scope = 'mmseg'
device = 'cuda'
dist_params = dict(backend='nccl')
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
find_unused_parameters = True
fp16 = dict()
gpu_multipliers = 1
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
launcher = 'pytorch'
load_from = None
log_config = dict(
    hooks=[
        dict(by_epoch=False, type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    by_epoch=False,
    min_lr=0.0,
    policy='poly',
    power=1.0,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-06)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=256,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=150,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        chunkwise_recurrents=[
            True,
            True,
            True,
            False,
        ],
        depths=[
            2,
            2,
            8,
            2,
        ],
        drop_path_rate=0.15,
        embed_dims=[
            64,
            128,
            256,
            512,
        ],
        heads_ranges=[
            4,
            4,
            6,
            6,
        ],
        init_cfg=dict(
            checkpoint=
            '/home/zengshimao/code/RMT/classfication_release/work_dirs/Unet/unet_t/best.pth',
            type='Pretrained'),
        init_values=[
            2,
            2,
            2,
            2,
        ],
        layer_init_values=1e-06,
        layerscales=[
            False,
            False,
            False,
            False,
        ],
        mlp_ratios=[
            3,
            3,
            3,
            3,
        ],
        num_heads=[
            4,
            4,
            8,
            16,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        topks=[
            1,
            4,
            16,
            -2,
        ],
        type='BiUnet'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            256,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=150,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=6e-05,
        type='AdamW',
        weight_decay=0.01),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=6e-05,
    momentum=0.9,
    type='AdamW',
    weight_decay=0.01)


optimizer_config = dict(loss_scale=512.0, type='Fp16OptimizerHook')
resume = False
resume_from = None
runner = dict(
    max_iters=160000,
    type='IterBasedRunner',
    work_dir='work_dirs/SegNet_Uper_s_2x')
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        data_root='data/ade/ADEChallengeData2016',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        flip=False,
        scale=(
            2048,
            512,
        ),
        transforms=[
            dict(keep_ratio=True, type='Resize'),
            dict(type='RandomFlip'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='MultiScaleFlipAug'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=8000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        data_root='data/ade/ADEChallengeData2016',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(ratio_range=(
        0.5,
        2.0,
    ), scale=(
        2048,
        512,
    ), type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(pad_val=0, size=(
        512,
        512,
    ), type='Pad'),
    dict(type='DefaultFormatBundle'),
    dict(keys=[
        'img',
        'gt_semantic_seg',
    ], type='Collect'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        data_root='data/ade/ADEChallengeData2016',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
work_dir = './work_dirs/BiUnet_Uper_t_2x'
workflow = [
    (
        'train',
        1,
    ),
]