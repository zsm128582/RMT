# from SegNet.segmentBackbone import VisSegNet_S
# from SegNet_conv.segmentBackbone import VisSegNet_conv_T
# from Qnet.segmentBackbone import Qnet_T
import torch
# import cv2
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import argparse
import matplotlib.pyplot as plt
import numpy as np
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

from tokenGalerkin_fixCollapes_v2.segmentBackbone import tokengalerkin_fixCollapse_t_v2


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = args.input_size if args.input_size > 224 else int((256 / 224) * args.input_size)
        # size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--early-conv', action='store_true')
    parser.add_argument('--conv-pos', action='store_true')
    parser.add_argument('--use-ortho', action='store_true')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')  
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--loadfrom', default='', help='load from checkpoint')
    parser.add_argument('--hook', default='', help='hook you')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def visualize_feature_maps(feature_maps):
    if feature_maps is None:
        return 
    feature_maps = feature_maps.permute(0,3,1,2)
    b , c , h , w = feature_maps.shape

    print(feature_maps.shape)
    if feature_maps.ndim == 4:
        feature_maps = feature_maps.squeeze(0)

    # 计算所有通道的平均值
    aggregated_map = torch.mean(feature_maps, dim=0)
    aggregated_map_np = aggregated_map.detach().cpu().numpy()

    # # 计算所有通道的最大值
    # aggregated_map = torch.max(feature_maps, dim=0)[0] # torch.max 返回值和索引，我们只需要值
    # aggregated_map_np = aggregated_map.detach().cpu().numpy()

    # 归一化到 [0, 1] 范围，以便于可视化
    min_val = np.min(aggregated_map_np)
    max_val = np.max(aggregated_map_np)
    normalized_map = (aggregated_map_np - min_val) / (max_val - min_val)

    # 使用 matplotlib 可视化
    plt.imshow(normalized_map, cmap='viridis') # 'viridis' 是一种常用的热力图颜色，magma' 也是一种很棒的热力图颜色

    plt.colorbar(label='Activation Intensity')
    plt.title('Mean Aggregation of Feature Map')
    plt.axis('off') # 不显示坐标轴
    plt.show()
    pass


features = None 
gradients = None
def forward_hook (module , input , output):
    global features
    x, _ = output
    features = x

def backward_hook(module, grad_input, grad_output):
    """
    grad_output 也是 tuple，对应 (∂L/∂x, ∂L/∂queries)
    我们只要 ∂L/∂x
    """
    global gradients
    if isinstance(grad_output, tuple):
        # grad_output 可能是 (grad_x, grad_queries) 或者只有一个
        grad_x = grad_output[0]
        grad_q = grad_output[1]
        print("gradient of q")
        print(grad_q)
    else:
        grad_x = grad_output
    gradients = grad_x
import torch.nn.functional as F


def visualize_attention_heatmap(
    attn_weights: torch.Tensor,
    original_image: torch.Tensor,
    h_feat: int,
    w_feat: int,
    alpha: float = 0.5,
    colormap: str = 'jet',
    save_path = None
):
    """
    可视化 Cross Attention 的完整热力图分布 (不使用 Top-K)。
    用于观察 Attention 是否弥散以及具体的关注强度分布。

    Args:
        attn_weights (torch.Tensor): Shape [1, h, q, n].
        original_image (torch.Tensor): Shape [1, 3, H, W].
        h_feat (int): 特征图高度.
        w_feat (int): 特征图宽度.
        alpha (float): 热力图透明度 (0.0 - 1.0).
        colormap (str): Matplotlib colormap.
    """
    
    # --- 1. 基础参数获取 ---
    b, q, n = attn_weights.shape
    _, _, H, W = original_image.shape
    
    # 移除 batch 维度
    attn = attn_weights[0] # [h, q, n]
    
    # --- 2. 降维聚合 ---
    # 对 heads 和 queries 取平均，得到全局的空间注意力分布
    # Shape: [n]
    attn_averaged = attn.mean(dim=0)
    
    # --- 3. 空间重塑 ---
    # [n] -> [h_feat, w_feat]
    attn_spatial = attn_averaged.view(h_feat, w_feat)
    
    # --- 4. 上采样到原图尺寸 ---
    # 增加维度以适配 interpolate: [1, 1, h_feat, w_feat]
    attn_spatial = attn_spatial.unsqueeze(0).unsqueeze(0)
    
    # 使用双线性插值 (bilinear) 使得热力图平滑过渡，这能更好体现分布趋势
    attn_upsampled = F.interpolate(
        attn_spatial, size=(H, W), mode='bilinear', align_corners=False
    )
    
    # 移除多余维度 -> [H, W]
    attention_map = attn_upsampled.squeeze()
    
    # --- 5. 归一化 (Normalization) ---
    # 为了让颜色能充分利用 colormap 的范围，我们需要将权重映射到 [0, 1]
    # 注意：这步操作对于观察"相对强弱"很有用，但会丢失"绝对数值"信息。
    # 如果所有权重都非常小（例如 1e-5），归一化后看起来也会很红。
    # 建议配合 print(attention_map.max()) 使用。
    attn_map_np = attention_map.detach().cpu().numpy()
    
    min_val, max_val,avg_val = attn_map_np.min(), attn_map_np.max(), np.mean(attn_map_np)
    print(f"[Debug] Attention Map Range: Min={min_val:.6f}, Max={max_val:.6f}, Avg={avg_val:.6f}" )
    
    if max_val - min_val > 1e-8:
        attn_norm = (attn_map_np - min_val) / (max_val - min_val)
    else:
        attn_norm = attn_map_np # 避免除以0，说明分布完全是平的
        
    # --- 6. 图像合成 ---
    # 6.1 准备原图
    img_display = original_image[0].clone().detach().cpu()
    # 简单的 Min-Max 归一化用于显示
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
    img_np = img_display.permute(1, 2, 0).numpy()
    
    # 6.2 生成热力图颜色
    cmap = plt.get_cmap(colormap)
    # cmap 返回 RGBA，取前3个通道
    heatmap_rgb = cmap(attn_norm)[:, :, :3]
    
    # 6.3 叠加
    # 公式: result = img * (1-alpha) + heatmap * alpha
    overlayed_img = img_np * (1 - alpha) + heatmap_rgb * alpha
    overlayed_img = np.clip(overlayed_img, 0.0, 1.0)
    
    # --- 7. 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 显示纯热力图 (使用伪彩色)
    im1 = axes[1].imshow(attn_norm, cmap=colormap)
    axes[1].set_title("Attention Heatmap (Raw Distribution)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04) # 添加色条
    
    axes[2].imshow(overlayed_img)
    axes[2].set_title(f"Overlay (alpha={alpha})")
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def visualize_cross_attention_topk(
    attn_weights: torch.Tensor,
    original_image: torch.Tensor,
    h_feat: int,
    w_feat: int,
    topk: int,
    alpha: float = 0.6,
    colormap: str = 'jet',
    save_path = None
):
    """
    可视化 Cross Attention 的 Top-K 激活区域并叠加在原图上。

    Args:
        attn_weights (torch.Tensor): Shape [b, h, q, n], 这里 b=1.
                                     n 必须等于 h_feat * w_feat.
        original_image (torch.Tensor): 原始图像, Shape [b, 3, H, W], 这里 b=1.
                                       假设已经 normalize 到 [0, 1] 或标准化过，
                                       我们会将其反标准化用于显示.
        h_feat (int): Key/Value 特征图的高度.
        w_feat (int): Key/Value 特征图的宽度.
        topk (int): 选取前多少个激活点.
        alpha (float): Heatmap 叠加的透明度 (0.0 - 1.0).
        colormap (str): Matplotlib 的 colormap 名称，例如 'jet', 'magma', 'viridis'.
    """
    
    # --- 1. 基础检查与参数获取 ---
    b, q, n = attn_weights.shape
    b_img, c_img, H, W = original_image.shape
    assert b == 1 and b_img == 1, "此脚本仅支持 batch size 为 1 的可视化"
    assert n == h_feat * w_feat, f"Attention dim n ({n}) 不等于 h_feat*w_feat ({h_feat}*{w_feat})"
    
    # 移除 batch 维度方便处理
    # attn = attn_weights[0] # [h, q, n]
    img_tensor = original_image[0] # [3, H, W]

    # --- 2. 降维聚合 ---
    # 你的设想：平均得到 [b, n] 的向量 (这里已经是 [n] 了)
    # 我们对 heads (dim 0) 和 queries (dim 1) 取平均
    # 含义：平均而言，哪些 key 的位置被 query 关注得最多
    attn_averaged = attn_weights.mean(dim=0).mean(dim=0) # Shape: [n]

    # --- 3. 选取 Top-K 并创建掩码 (在特征图层级) ---
    # 找到 topk 的值和索引
    # values, indices = torch.topk(attn_averaged, topk) 
    # 我们只需要索引来创建 mask
    _, topk_indices = torch.topk(attn_averaged, topk)
    
    # 创建一个全零的 mask [n]
    mask_flat = torch.zeros_like(attn_averaged)
    # 将 topk 位置置为 1
    mask_flat[topk_indices] = 1.0
    
    # --- 4. 空间重塑与上采样 ---
    # 将平铺的 mask 重塑为特征图空间维度 [h_feat, w_feat]
    mask_feat_map = mask_flat.view(h_feat, w_feat)
    
    # 为了使用 interpolate，需要增加 batch 和 channel 维度 -> [1, 1, h_feat, w_feat]
    mask_feat_map_bc = mask_feat_map.unsqueeze(0).unsqueeze(0)
    
    # 上采样到原始图像尺寸 [H, W]
    # 使用 'nearest' 可以看到清晰的块状 topk 区域 (忠实还原特征图上的选择)
    # 使用 'bilinear' 可以看到平滑的区域 (视觉效果更好)
    # 这里我们用 bilinear 使得叠加效果更平滑自然
    mask_upsampled = F.interpolate(
        mask_feat_map_bc, size=(H, W), mode='bilinear', align_corners=False
    )
    
    # 移除多余维度 -> [H, W]
    attention_map = mask_upsampled.squeeze()
    
    # --- 5. 可视化准备 ---
    
    # 5.1 处理原始图像用于显示
    # 假设输入图像已经标准化过，我们需要将其还原到 [0, 1] 区间用于 matplotlib 显示
    # 这里做一个简单的 min-max normalization 来还原，实际情况请根据你的预处理方式调整
    img_display = img_tensor.clone().detach().cpu()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
    # 转为 [H, W, 3] 的 numpy 数组
    img_np = img_display.permute(1, 2, 0).numpy()

    # 5.2 处理 Attention Map 用于显示
    attn_map_np = attention_map.detach().cpu().numpy()
    # 归一化到 [0, 1]
    attn_map_np = (attn_map_np - attn_map_np.min()) / (attn_map_np.max() - attn_map_np.min() + 1e-8)

    # 应用 colormap 制作 heatmap
    cmap = plt.get_cmap(colormap)
    # cmap(attn_map_np) 会返回 [H, W, 4] (RGBA)，我们取前3个通道 RGB，并且舍弃 alpha 通道
    heatmap = cmap(attn_map_np)[:, :, :3] 

    # --- 6. 叠加与绘图 ---
    # 叠加公式: output = image * (1-alpha) + heatmap * alpha
    # 这里的 heatmap 实际上是根据 topk 生成的 mask 的热力图表现
    overlayed_img = img_np * (1 - alpha) + heatmap * alpha
    # 确保数值范围安全
    overlayed_img = np.clip(overlayed_img, 0.0, 1.0)

    # 创建绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image ({H}x{W})")
    axes[0].axis('off')
    
    # 显示上采样后的 Top-K Mask (黑白)
    axes[1].imshow(attn_map_np, cmap='gray')
    axes[1].set_title(f"Top-{topk} Attention Mask (Upsampled)")
    axes[1].axis('off')
    
    # 显示叠加结果
    axes[2].imshow(overlayed_img)
    axes[2].set_title(f"Overlay (Top-{topk}, alpha={alpha})")
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

from pathlib import Path
def get_attention_weight(original_image ,save_path):
    def visualHook(module , input , output):
        # queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor , h , w
        _,_,_,_,h ,w = input
        # attention weight  : b h q N  -> b N  - > b topk  
        _ , _ , attn_weights = output
        # print(attn_weights.shape)
        b, q, n = attn_weights.shape
        p = Path(save_path)
        name = p.name
        basePath = Path(p.parent)

        for i in range(q):
            q_save_path = basePath / f"q{i}" 
            q_save_path.mkdir(parents=True , exist_ok=True)
            q_save_path = q_save_path / name
            # visualize_cross_attention_topk(attn_weights,original_image,h,w,(int)(h*w*0.5),save_path=save_path)
            visualize_attention_heatmap(attn_weights[:,i,:].unsqueeze(1),original_image,h,w,save_path=q_save_path)
    return visualHook


import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    work_dir = "/home/u2023110769/code/RMT/classfication_release/work_dirs/tokengalerkin_v2_full_pre"
    resume = os.path.join(work_dir , "checkpoint.pth")
    
    checkpoint = torch.load(resume, map_location='cpu',weights_only=False)
    args.nb_classes = 1000
    model = tokengalerkin_fixCollapse_t_v2(args)
    depths=[2,2,8,2]
    # model = VisSegNet_argmax_S(args)
    # depths = [3, 4, 18, 4]
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    preprocess = build_transform(False, args)
    images = {
        "dog1" : "/home/u2023110769/datasets/ImageNet1k/val/n02113712/n02113712_43516.JPEG",
        "dog2" : "/home/u2023110769/datasets/ImageNet1k/val/n02113712/n02113712_10575.JPEG",
        "fish" : "/home/u2023110769/datasets/ImageNet1k/val/n01440764/n01440764_2138.JPEG",
        "bird" : "/home/u2023110769/datasets/ImageNet1k/val/n01440764/n01440764_10306.JPEG"
    }
    for name , image in images.items():
        visualize_save = os.path.join(work_dir , "mask_visualization", name)
        if not os.path.exists(visualize_save):
            os.makedirs(visualize_save)
        img = Image.open(image).convert("RGB")
        img.save(os.path.join(visualize_save , "image.png"))

        input_tensor = preprocess(img).unsqueeze(0)
        hooks = []
        for layer_index in range(0,4):
            for block_index in range(depths[layer_index]):
                hooks.append( model.layers[layer_index].blocks[block_index].register_forward_hook(get_attention_weight(input_tensor,os.path.join(visualize_save , f"layer_{layer_index}_block_{block_index}.png"))))
        with torch.no_grad():
            res = model(input_tensor,200)
        for h in hooks:
            h.remove()