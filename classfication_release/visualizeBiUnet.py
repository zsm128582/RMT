# from SegNet.segmentBackbone import VisSegNet_S
# from SegNet_conv.segmentBackbone import VisSegNet_conv_T
# # from Biformer.Biformer import biformer_tiny
from UNet.BiUnet import BiUnet_t
import torch
import cv2
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import argparse
import matplotlib.pyplot as plt
import numpy as np



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

mask = None
def getMask(layer , block):
    def calHook(module , input , output):
        # print(input) # 输出为()
        # print(module) # 输出正确
        # print(output) # 存在输出
        _ , _ , attentionMap , _ , _ = input  # ValueError: not enough values to unpack (expected 4, got 0)
        global mask
        mask = attentionMap.detach()
        # with torch.no_grad():
        #     token_embeddings = queries.squeeze(0) 
        #     norm_embeddings = F.normalize(token_embeddings, p=2, dim=1)
        #     similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.transpose(0, 1))
        #     print("--------"*20)
        #     print(f"layer :{layer} - block: {block}")
        #     print(f"mean similarity:{similarity_matrix.mean()}")
        #     print(similarity_matrix)
    return calHook


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# # --- 1. 模拟数据 (替换为你的实际数据) ---

# # 原始图像 (假设是灰度图，如果彩色图需要修改)
# # 这里创建一个模拟的224x224的图像
# original_image_path = "path/to/your/original_image.jpg" # 替换为你的图像路径
# try:
#     original_image = Image.open(original_image_path).convert('RGB')
#     original_image_np = np.array(original_image)
# except FileNotFoundError:
#     print("Warning: Original image not found. Creating a dummy image.")
#     original_image_np = np.random.rand(224, 224, 3) * 255
#     original_image_np = original_image_np.astype(np.uint8)

# # 模拟的注意力分配 [1, 49, 49]
# # 确保注意力是正值且行和为1 (如果需要)
# attention_map_np = np.random.rand(49, 49)
# # 对每行进行归一化，使其和为1，模拟softmax后的结果
# attention_map_np = attention_map_np / attention_map_np.sum(axis=1, keepdims=True)
# attention_map_np = attention_map_np[np.newaxis, :, :] # 增加batch维度 [1, 49, 49]

# 定义参数
image_h, image_w = 224, 224 # 原始图像尺寸
feature_h, feature_w = 56, 56 # hook得到的特征图尺寸
window_size = 8 # p=8, 粗糙token窗口大小
num_regions_h = image_h // window_size # 224 // 8 = 28
num_regions_w = image_w // window_size # 224 // 8 = 28
# 注意: 你的描述中提到 "粗糙token为49个"，这意味着 7x7 的粗糙token网格，而不是 28x28。
# 如果是 7x7，则每个粗糙token对应的原始图像区域是 32x32。
# 让我们根据你的"49个粗糙token"来调整参数:
num_coarse_tokens = 49
regions_per_dim = int(np.sqrt(num_coarse_tokens)) # 7
region_pixel_size = image_h // regions_per_dim # 224 // 7 = 32

print(f"原始图像尺寸: {image_h}x{image_w}")
print(f"粗糙token网格: {regions_per_dim}x{regions_per_dim} ({num_coarse_tokens} tokens)")
print(f"每个粗糙token对应的原始图像区域尺寸: {region_pixel_size}x{region_pixel_size}")


# --- 2. 辅助函数 ---

def get_region_coords(region_idx, regions_per_dim, region_pixel_size):
    """
    根据粗糙token的索引获取其在原始图像上的像素坐标 (左上角x, 左上角y, 宽度, 高度)
    """
    row = region_idx // regions_per_dim
    col = region_idx % regions_per_dim
    x_start = col * region_pixel_size
    y_start = row * region_pixel_size
    return x_start, y_start, region_pixel_size, region_pixel_size

def get_region_idx_from_coords(px, py, regions_per_dim, region_pixel_size):
    """
    根据点击的像素坐标获取对应的粗糙token索引
    """
    if not (0 <= px < image_w and 0 <= py < image_h):
        return -1 # 超出图像范围

    col = px // region_pixel_size
    row = py // region_pixel_size
    return row * regions_per_dim + col

# --- 3. 可视化函数 ---

def visualize_attention(original_image_np, attention_map_np, top_k=5):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(original_image_np)
    ax.set_title(f"Click a region to visualize its top {top_k} attended regions")
    ax.axis('off')

    selected_region_idx = -1

    def onclick(event):
        nonlocal selected_region_idx
        if event.inaxes != ax:
            return

        x, y = int(event.xdata), int(event.ydata)
        clicked_region_idx = get_region_idx_from_coords(x, y, regions_per_dim, region_pixel_size)

        if clicked_region_idx != -1 and clicked_region_idx != selected_region_idx:
            selected_region_idx = clicked_region_idx
            print(f"Clicked pixel: ({x}, {y}), corresponding to region index: {selected_region_idx}")
            update_visualization(selected_region_idx)

    def update_visualization(current_selected_idx):
        # 清除之前的矩形框
        for p in reversed(ax.patches):
            p.remove()

        # 绘制所有区域的网格 (可选，可以帮助理解划分)
        # for i in range(num_coarse_tokens):
        #     x, y, w, h = get_region_coords(i, regions_per_dim, region_pixel_size)
        #     rect = patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor='gray', facecolor='none', alpha=0.3)
        #     ax.add_patch(rect)


        # 1. 绘制被选中的区域
        x_sel, y_sel, w_sel, h_sel = get_region_coords(current_selected_idx, regions_per_dim, region_pixel_size)
        rect_sel = patches.Rectangle((x_sel, y_sel), w_sel, h_sel, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='Selected Region')
        ax.add_patch(rect_sel)

        # 2. 获取并绘制top k最相关的区域
        # 注意力分配是 [1, 49, 49]，所以我们取第0个batch的第current_selected_idx行
        current_attention_scores = attention_map_np[0, current_selected_idx, :]

        # 获取top k的索引
        top_k_indices = np.argsort(current_attention_scores)[::-1][1:top_k+1] # [1:]排除自身

        print(f"Top {top_k} attended regions for region {current_selected_idx}: {top_k_indices.tolist()}")

        for i, idx in enumerate(top_k_indices):
            x, y, w, h = get_region_coords(idx, regions_per_dim, region_pixel_size)
            # 颜色可以根据注意力强度来调整，这里简单用蓝色
            alpha_val = 0.2 + 0.6 * (current_attention_scores[idx] / current_attention_scores[top_k_indices[0]]) # 越高亮越强
            rect_attn = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='blue', facecolor='blue', alpha=alpha_val, label=f'Attended Region {i+1}')
            ax.add_patch(rect_attn)

        fig.canvas.draw_idle()

    # 连接点击事件
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    resume = "/home/zengshimao/code/RMT/classfication_release/work_dirs/Unet/unet_t/best.pth"
    checkpoint = torch.load(resume, map_location='cpu',weights_only=False)
    args.nb_classes = 100
    model = BiUnet_t(args)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.cuda()
    model.eval()

    # preprocess = transforms.Compose([
    # transforms.Resize((224,224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                      std=[0.229, 0.224, 0.225])
    # ])
    preprocess = build_transform(False, args)


    # 小狗：/home/zengshimao/datasets/ImageNet1k/val/n02113712/n02113712_43516.JPEG
    # 狗2 ：/home/zengshimao/datasets/ImageNet1k/val/n02113712/n02113712_10575.JPEG
    # 鱼：/home/zengshimao/datasets/ImageNet1k/val/n01440764/n01440764_2138.JPEG
    # img = Image.open("/home/zengshimao/datasets/ImageNet1k/val/n01440764/n01440764_10306.JPEG").convert("RGB")
    img = Image.open("/home/zengshimao/datasets/ImageNet1k/train/n02110806/n02110806_14081.JPEG").convert("RGB")
    

    input_tensor = preprocess(img).unsqueeze(0)
    input_tensor = input_tensor.cuda()
    # 这里需要把transform后的图片保存一下
    hooks = []
    hooks.append(model.layers[1].register_forward_hook(getMask(1,0)))

    # for layer_index in range(1,4):
    #     for block_index in range(depths[layer_index]):
    #         hooks.append( model.layers[layer_index].blocks[block_index].register_forward_hook(calQueriesSimilarity(layer_index , block_index)))


    with torch.no_grad():
        res = model(input_tensor)

    print(mask.shape)
    for h in hooks:
        h.remove()
        # --- 运行可视化 ---
    visualize_attention(input_tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy(), mask.detach().cpu().numpy(), top_k=5)


    #B,H,W,C = x.shape

""""
可视化方法：
import matplotlib.pyplot as plt
region_id = mask_logits.argmax(dim=-1)
region_map = region_id.reshape(28,28)
cmap = plt.cm.get_cmap('tab10', 8)
plt.imshow(region_map, cmap=cmap)
plt.savefig("/home/zengshimao/code/RMT/classfication_release/work_dirs/SegNet/inference/7*7mask.png")
"""