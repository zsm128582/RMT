_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        _delete_ = True,
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
        drop_path_rate=0.1,
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
            "/home/u2023110769/code/RMT/detect_mmengine/workdirs/tokenGalerkinV2_full_200e.pth",
            type='Pretrained'),
        init_values=[
            2,
            2,
            2,
            2,
        ],
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
        type='tokenNet'),
    neck=dict(in_channels=[64, 128, 256, 512])
)

# train_dataloader = dict(batch_size=2) # as gpus=8

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

