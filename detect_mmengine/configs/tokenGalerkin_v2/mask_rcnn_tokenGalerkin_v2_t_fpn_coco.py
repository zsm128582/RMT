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

# too big
# train_dataloader = dict(batch_size=1) # as gpus=16



