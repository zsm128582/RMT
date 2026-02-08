_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

auto_scale_lr = dict(base_batch_size=16, enable=True)

model = dict(
    backbone=dict(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False],
        out_indices = (0, 1, 2, 3),
        init_cfg=dict(
            checkpoint=
            "/home/u2023110769/code/RMT/detect_mmengine/workdirs/tokenGalerkinV2_full_200e.pth",
            type='Pretrained'),
        type='RMT',
    ),
    neck=dict(in_channels=[64, 128, 256, 512])
)


optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')

# too big
train_dataloader = dict(batch_size=4) # as gpus=16



