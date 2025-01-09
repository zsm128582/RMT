_base_ = [
    "../_base_/models/TNT_upper.py",
    "../_base_/datasets/ade20k_uper.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

import torch.nn as nn
model = dict(
    pretrained=None,
        backbone=dict(
        type="TNT",
        img_size=512,#??
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
        in_channels=[384, 384, 384, 384],
        num_classes=150,
        channels=384,
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150
    ), 
)

############## below we strictly follow uniformer & cswin ####################################
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/exp/upernet_global_small/config.py
# https://github.com/microsoft/CSWin-Transformer/blob/main/segmentation/configs/cswin/upernet_cswin_tiny.py
##############################################################################################
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone

gpu_multipliers = 1

optimizer = dict(
    _delete_=True,
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
lr_config = dict(
    _delete_=True,
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
runner = dict(max_iters=160000 // (gpu_multipliers), work_dir="path/RMT_Uper_s_2x")
checkpoint_config = dict(max_keep_ckpts=1, interval=8000 // (gpu_multipliers))
evaluation = dict(interval=8000 // (gpu_multipliers), save_best="mIoU")

# NOTE: True is conflict with checkpoint
# https://github.com/allenai/longformer/issues/63#issuecomment-648861503
# What the fuck ?
find_unused_parameters = True

# place holder for new verison mmseg compatiability
resume_from = None
device = "cuda"

# fp32 training (choose this if nan loss occurs)->
# optimizer_config = dict()

# AMP (faster but may meet nan loss) ->
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
fp16 = dict()
