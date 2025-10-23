_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
# load_from="/home/zengshimao/code/RMT/classfication_release/work_dirs/Unet/unet_t/best.pth"
gpu_multipliers=1
default_scope = 'mmseg'
model = dict(
    pretrained = None,
    # pretrained="/home/zengshimao/code/RMT/classfication_release/work_dirs/Unet/unet_t/best.pth",

    backbone=dict(
        _delete_=True,
        init_cfg=dict(type='Pretrained', 
                      checkpoint='/home/zengshimao/code/RMT/classfication_release/work_dirs/Unet/unet_t/best.pth'),
        # pretrained="/home/zengshimao/code/RMT/classfication_release/work_dirs/Unet/unet_t/best.pth",

        # pretrained=None,
        type='BiUnet',
        embed_dims=[64, 128, 256, 512],
        topks=[1, 4, 16, -2],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False],
        out_indices=(0, 1, 2, 3),
    ),  # it seems that, upernet requires a larger dpr
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150),
    )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)


optimizer = dict(
    type="AdamW",
    lr=0.00006 * gpu_multipliers,
    betas=(0.9, 0.999),
    weight_decay=0.01,

)

optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
fp16 = dict()

val_cfg = dict(type='ValLoop')
