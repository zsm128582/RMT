import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from typing import Tuple, Union
from functools import partial
from einops import einsum
from einops import rearrange

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class DownSample(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
        )
    def forward(self , x ):
        x = self.proj(x)
        return x
    
class Encoder(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=[64, 128, 256, 512], norm_layer=None ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.ModuleList()
        self.proj.append(DownSample(in_chans ,  embed_dim[0]//2))
        self.proj.append(DownSample(embed_dim[0]//2 ,  embed_dim[0]))
        self.proj.append(DownSample(embed_dim[0] , embed_dim[1]))
        self.proj.append(DownSample(embed_dim[1] ,  embed_dim[2]))
        self.proj.append(DownSample(embed_dim[2] ,  embed_dim[3]))
        self.activate = nn.GELU()

    def forward(self, x):
        #input bchw
        features = []
        x = self.proj[0](x)
        for i in range(len(self.embed_dim)):
            x = self.activate(x)
            x = self.proj[i+1](x)
            features.append(x.permute(0,2,3,1))
        # return bhwc
        # featrues : 1/4 , 1/8 , 1/16 , 1/32 
        x = x.permute(0,2,3,1)
        # 忽略1/32  ， 并倒序 ， 获得None ， 16，8，4
        features = features[-2::-1]
        features.insert(0,None)
        return x , features
    