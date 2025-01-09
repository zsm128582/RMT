# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models import register

from galerkin_transformer.model import SimpleAttention
from .positionEmbedding import PositionEmbeddingSine
from .SimpleGalerkin import SelfAttention

# from pos_embed import get_2d_sincos_pos_embed


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
        image_size=224,
        patch_size=16,
        img_chans=64,
        embed_dim=1024,
    ):

        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        # self.patch_embed = PatchEmbed(image_size, patch_size, img_chans, embed_dim)
        # num_patches = self.patch_embed.num_patches
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # self.rebuild = nn.Linear(embed_dim, patch_size** 2 * img_chans, bias=True) # decoder to patch

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res
        # # 1. image patchfy (2d convolution)
        # res = self.patch_embed(res)

        # #  res.shape  [ b , patch_num , emdding dimension]  [16 , 196  ,1024]
        # # 2. pos embedding
        # res = res + self.pos_embed[:,1,:]

        # # 3. galerkin

        # res , _ = self.sa1(query = res , key = res , value = res)

        # # 4. rebuild or reshape ? [ b , 196 , 16384]

        # res = self.rebuild(res)

        # # [ b , 196 , 16384] ->  [b , 64 ,  224 , 224 ]
        # res = self.unpatchify(res , img_chans= self.img_chans)

        # TODO : 尝试用patch 吧
        # pos = self.posEmbedding(res, None)
        # res , _ = self.sa1(query = res , key = res , value = res , pos = pos)


    # 1. 在最后resblock加gk
    # 2. 减少维度（
    # def unpatchify(self, x, img_chans=3):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1] ** 0.5)
    #     assert h * w == x.shape[1]
    #     # 14 ,14 , 16 ,16  , 64
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, img_chans))
    #     x = torch.einsum("nhwpqc->nchpwq", x)
    #     # 64 , 14 * 16  , 14 *16  ,
    #     imgs = x.reshape(shape=(x.shape[0], img_chans, h * p, h * p))
    #     return imgs


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    "r16f64x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
    "r16f64x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt",
    "r16f64x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
    "r32f256x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt",
    "r32f256x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt",
    "r32f256x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt",
}


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = "r{}f{}x{}".format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]


        m_body = []

        # m_body.append(SelfAttention(n_feats))
        for _ in range(n_resblocks):
            m_body.append(ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        # define body module
        # m_body = [
        #     ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
        #     for _ in range(n_resblocks)
        # ]
        


        # new_body = []
        # for i, block in enumerate(m_body):
        #     new_body.append(block)
        #     if( (i+1) % 4 == 0):
        #         new_body.append(SelfAttention(n_feats))

        # new_body.append(conv(n_feats, n_feats, kernel_size))  
        # del m_body


        # self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*new_body)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        # print(self.body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size),
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        # x
        res += x

        # 配置文件中写的是no upsampling 所以后面的可以不用管了
        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)

        # x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))



def make_edsr_baseline(
    n_resblocks=16, n_feats=256, res_scale=1, scale=2, no_upsampling=False, rgb_range=1
):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)



def make_edsr(
    n_resblocks=32,
    n_feats=256,
    res_scale=0.1,
    scale=2,
    no_upsampling=False,
    rgb_range=1,
):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)

