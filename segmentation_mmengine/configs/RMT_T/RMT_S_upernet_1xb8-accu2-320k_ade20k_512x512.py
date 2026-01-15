_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
train_dataloader = dict(
    batch_size=8,
)

optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    accumulative_counts=2,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')

param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=3000, start_factor=1e-06, # 这里多了500轮warmup 不过应该无伤大雅
        type='LinearLR'),
    dict(
        begin=3000,
        by_epoch=False,
        end=320000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]

train_cfg = dict(
    max_iters=320000, type='IterBasedTrainLoop', val_interval=32000)

checkpoint="/home/u2023110769/code/RMT/segmentation_mmengine/checkpoint/RMT-S.pth"
model = dict(
    backbone=dict(
        _delete_ = True,
                embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False],
        init_cfg=dict(
            checkpoint=checkpoint,
            type='Pretrained'),
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        type='VisRetNet'),
        decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
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
)
