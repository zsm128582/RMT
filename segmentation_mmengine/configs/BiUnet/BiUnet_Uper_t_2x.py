_base_ = [
    "../_base_/models/BiUnet_t_upper.py",
    "../_base_/datasets/ade20k_uper.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

# model.pretrained is actually loaded by backbone, see
# https://github.com/open-mmlab/mmsegmentation/blob/186572a3ce64ac9b6b37e66d58c76515000c3280/mmseg/models/segmentors/encoder_decoder.py#L32

model = dict(
    pretrained=None,
    
    backbone=dict(
        type='BiUnet',
                init_cfg=dict(type='Pretrained', 
                      checkpoint='/home/zengshimao/code/RMT/classfication_release/work_dirs/Unet/unet_t/best.pth'),
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


############## below we strictly follow uniformer & cswin ####################################
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/exp/upernet_global_small/config.py
# https://github.com/microsoft/CSWin-Transformer/blob/main/segmentation/configs/cswin/upernet_cswin_tiny.py
##############################################################################################
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone

gpu_multipliers = 1


lr_config = dict(

    policy="poly",
    warmup="linear",
    # warmup_iters=1500 // (gpu_multipliers // 2),
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
#############################################################################
# runner = dict(max_iters=160000//(gpu_multipliers//2), work_dir='path\/RMT_Uper_s_1x')
runner = dict(max_iters=160000 // (gpu_multipliers), work_dir="work_dirs/SegNet_Uper_s_2x")
checkpoint_config = dict(max_keep_ckpts=1, interval=8000 // (gpu_multipliers))
evaluation = dict(interval=8000 // (gpu_multipliers), save_best="mIoU")

# NOTE: True is conflict with checkpoint
# https://github.com/allenai/longformer/issues/63#issuecomment-648861503
find_unused_parameters = True

# place holder for new verison mmseg compatiability
resume_from = None
device = "cuda"

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
# fp32 training (choose this if nan loss occurs)->
# optimizer_config = dict()

# AMP (faster but may meet nan loss) ->
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
fp16 = dict()

optimizer = dict(

    type="AdamW",
    lr=0.00006 * gpu_multipliers,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
val_cfg = dict(type='ValLoop')

