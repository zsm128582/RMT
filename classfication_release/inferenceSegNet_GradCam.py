from SegNet.segmentBackbone import VisSegNet_S
from SegNet_conv.segmentBackbone import VisSegNet_conv_T
from Qnet.segmentBackbone import Qnet_T
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


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
    else:
        grad_x = grad_output
    gradients = grad_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    resume = "/home/zengshimao/code/RMT/classfication_release/work_dirs/SegNet/QNet/best.pth"
    checkpoint = torch.load(resume, map_location='cpu',weights_only=False)
    args.nb_classes = 1000
    model = Qnet_T(args)

    target_layer = model.layers[3]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()


    #归一化
    transform = build_transform(False, args)
    root = "/home/zengshimao/datasets/ImageNet1k/val/"
    # dataset = datasets.ImageFolder(root, transform=transform)
    # testImage , target = dataset.__getitem__(0)

    # testImage = torch.unsqueeze(testImage, 0)
    preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    def grad_cam(imagepath , savepath):
        img = Image.open(imagepath).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)


        res = model(input_tensor,200)
        pred_class = res.argmax(dim = -1)

        model.zero_grad()
        res[0,pred_class].backward()
        global gradients
        global features

        gradients = gradients.permute(0,3,1,2)
        features = features.permute(0,3,1,2)

        weights = gradients.mean(dim=[2, 3], keepdim=True)  
        cam = (weights * features).sum(dim=1, keepdim=True)


        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=(224,224), mode='bilinear', align_corners=False)
        
        cam = cam.squeeze().detach().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min())
        heatmap = plt.cm.jet(cam)[..., :3]
        overlay = 0.5*heatmap + 0.5*np.array(img.resize((224,224)))/255
        print(overlay.shape)

        plt.imshow(overlay)

        plt.axis("off")
        plt.savefig(savepath)
    
    # imagepaths = [
    #     "/home/zengshimao/datasets/ImageNet1k/val/n02113712/n02113712_43516.JPEG",
    #     "/home/zengshimao/datasets/ImageNet1k/val/n02113712/n02113712_10575.JPEG",
    #     "/home/zengshimao/datasets/ImageNet1k/val/n01440764/n01440764_2138.JPEG"
    # ]
    # for index , path in enumerate(imagepaths):
    #     grad_cam(path , f"/home/zengshimao/code/RMT/classfication_release/work_dirs/SegNet/QNet/grad_cam_{index}.png")
    grad_cam("/home/zengshimao/datasets/ImageNet1k/val/n01440764/n01440764_10306.JPEG" , "/home/zengshimao/code/RMT/classfication_release/work_dirs/SegNet/QNet/grad_cam.png")

    # resume = "/home/zengshimao/code/RMT/classfication_release/work_dirs/SegNet/gumbel-softmax/best.pth"
    # checkpoint = torch.load(resume, map_location='cpu',weights_only=False)
    # model = VisSegNet_S(None)
    



    # feature_maps = None 
    # queries = None
    # def hook_fn(module , input , output):
    #     global feature_maps
    #     feature_maps , queries = output
        
        
    # target_layer = dict(model.named_modules())["layers.0"]
    # hook = target_layer.register_forward_hook(hook_fn)




    # model.load_state_dict(checkpoint['model'], strict=False)
    # model.eval()



    # #归一化
    # transform = build_transform(False, args)
    # root = "/home/zengshimao/datasets/ImageNet1k/val/"
    # dataset = datasets.ImageFolder(root, transform=transform)
    # testImage , target = dataset.__getitem__(0)
    # testImage = torch.unsqueeze(testImage, 0)
    # with torch.no_grad():
    #     res = model(testImage,200)
    # hook.remove()
    # print(res)

    # visualize_feature_maps(feature_maps)


    

""""
可视化方法：
import matplotlib.pyplot as plt
region_id = mask_logits.argmax(dim=-1)
region_map = region_id.reshape(28,28)
cmap = plt.cm.get_cmap('tab10', 8)
plt.imshow(region_map, cmap=cmap)
plt.savefig("/home/zengshimao/code/RMT/classfication_release/work_dirs/SegNet/inference/7*7mask.png")
"""