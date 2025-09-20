_base_ = [
    '../_base_/models/SegNet_conv_fpn.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# model.pretrained is actually loaded by backbone, see
# https://github.com/open-mmlab/mmsegmentation/blob/186572a3ce64ac9b6b37e66d58c76515000c3280/mmseg/models/segmentors/encoder_decoder.py#L32

# model = dict(
#     pretrained=None,
#     backbone=dict(
#         embed_dims=[64, 128, 256, 512],
#         depths=[3, 4, 18, 4],
#         num_heads=[4, 4, 8, 16],
#         init_values=[2, 2, 2, 2],
#         heads_ranges=[4, 4, 6, 6],
#         mlp_ratios=[4, 4, 3, 3],
#         drop_path_rate=0.15,
#         chunkwise_recurrents=[True, True, True, False],
#         layerscales=[False, False, False, False],
#         out_indices=(0, 1, 2, 3),
#     ),  # it seems that, upernet requires a larger dpr
#     decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
#     auxiliary_head=dict(in_channels=256, num_classes=150),
# )

model = dict(
    type='EncoderDecoder',
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
        out_indices = (0, 1, 2, 3)
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))



############## below we strictly follow uniformer & cswin ####################################
# https://github.com/Sense-X/UniFormer/blob/main/semantic_segmentation/exp/upernet_global_small/config.py
# https://github.com/microsoft/CSWin-Transformer/blob/main/segmentation/configs/cswin/upernet_cswin_tiny.py
##############################################################################################
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone

gpu_multipliers = 1

optimizer = dict(type='AdamW', lr=0.0001*gpu_multipliers, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
#############################################################################
# runner = dict(max_iters=160000//(gpu_multipliers//2), work_dir='path\/RMT_Uper_s_1x')
runner = dict(max_iters=80000, work_dir="work_dirs/VSN/conv/SegNet_conv_fpn_t_1x")
checkpoint_config = dict(interval=4000)
evaluation = dict(interval=4000)

# NOTE: True is conflict with checkpoint
# https://github.com/allenai/longformer/issues/63#issuecomment-648861503
find_unused_parameters = True

# place holder for new verison mmseg compatiability
resume_from = None
device = "cuda"

# fp32 training (choose this if nan loss occurs)->
# optimizer_config = dict()

# AMP (faster but may meet nan loss) ->
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
fp16 = dict()
