_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint="/home/zengshimao/code/RMT/segmentation_mmengine/checkpoint/agent_swin_t.pth"
train_dataloader = dict(
    batch_size=4,
    sampler=dict(shuffle=False, type='InfiniteSampler')
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
    max_iters=320000, type='IterBasedTrainLoop', val_interval=16000)
model = dict(
    backbone=dict(
        _delete_ = True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        type='AgentSwinTransformer',
        img_size=224, 
        patch_size=4, 
        in_chans=3,
        num_classes=150,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=56,
        mlp_ratio=4,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes = [(9, 9), (12, 12), (14, 14), (7, 7)],
        kernel_size=3, 
        attn_type='AAAB',
        scale=-0.5,
        ),
        
        decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
        auxiliary_head=dict(
            align_corners=False,
            channels=256,
            concat_input=False,
            dropout_ratio=0.1,
            in_channels=384,
            in_index=2,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=150,
            num_convs=1,
            type='FCNHead'),
)

