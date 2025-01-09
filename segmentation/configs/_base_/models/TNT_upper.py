# copied from mmsegmentation official config
# https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/upernet_r50.py

# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="TNT",
        img_size=224,#??
        patch_size=16,
        in_chans=32,
        num_classes=1000,
        outer_dim=384,
        inner_dim=24,
        depth=12,
        outer_num_heads=6,
        inner_num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        inner_stride=4,
        se=0,
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[80, 160, 320, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
