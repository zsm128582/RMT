
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
from .modules import Encoder,PatchExpand

from einops import rearrange


import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention, create_block_mask
)
flex_attention = torch.compile(flex_attention, dynamic=True)
torch._dynamo.config.cache_size_limit = 192
torch._dynamo.config.accumulated_cache_size_limit = 192

class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None, side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5


        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        self.project = nn.Linear(dim , 3*dim)
        self.wo = nn.Linear(dim , dim)
        
    def qkv(self , x):
        q , k, v = self.project(x).split([self.dim , self.dim , self.dim] , dim = -1)
        return q, k,v

    def forward(self, x, clstoken , mask):
        """
        x: NHWC tensor
        clstoken : N Q C

        Return:
            NHWC tensor
        """
        B , H , W , C = x.shape
        _, num_q ,_ = clstoken.shape
        h_w = H // self.n_win

        # 首先先做了linear ， 变成q k v : b h w c 
        q , k , v = self.qkv(x)

        q_cls , k_cls , v_cls = self.qkv(clstoken)

        q_cls = rearrange(q_cls , 'b q (Nh Hd) -> b Nh q Hd', Nh = self.num_heads)
        k_cls = rearrange(k_cls , 'b q (Nh Hd) -> b Nh q Hd', Nh = self.num_heads)
        v_cls = rearrange(v_cls , 'b q (Nh Hd) -> b Nh q Hd', Nh = self.num_heads)

        # b c h w
        lepe = self.lepe(rearrange(v , 'b h w c -> b c h w').contiguous())
        lepe = rearrange(lepe , 'b c h w -> b h w c')


        # b (j h ) ( i w ) (Nh Hd)  -> b Nh (j i h w) Hd
        q = rearrange(q,'b (j h ) ( i w ) (Nh Hd)  -> b Nh (j i h w) Hd' , j = self.n_win , i=self.n_win , Nh = self.num_heads)
        k = rearrange(k,'b (j h ) ( i w ) (Nh Hd)  -> b Nh (j i h w) Hd' , j = self.n_win , i=self.n_win , Nh = self.num_heads)
        v = rearrange(v,'b (j h ) ( i w ) (Nh Hd)  -> b Nh (j i h w) Hd' , j = self.n_win , i=self.n_win , Nh = self.num_heads)

        
        q_with_cls = torch.cat((q , q_cls) , dim=2)
        k_with_cls = torch.cat((k , k_cls) , dim=2)
        v_with_cls = torch.cat((v , v_cls) , dim=2)

        def mask_mod(b , h , q_idx , kv_idx):
            # return (q_idx >= H*W) | (kv_idx >= H*W) | (mask[b ,q_idx - numq//(h_w * h_w) , kv_idx//(h_w * h_w) ]==1)
            #因为不能使用逻辑语句， 所以只能在mask边缘添加一圈表示true ， 所以cls token目前只能使用一个了。
            return mask[b ,q_idx //(h_w * h_w) , kv_idx//(h_w * h_w) ]==1 
        
        blockmask = create_block_mask(mask_mod , B = B , H = self.num_heads , Q_LEN= H*W+num_q , KV_LEN=H*W+num_q ,_compile=True)

        
        #x : b H  (j i h w + num_q) c
        out = flex_attention(q_with_cls , k_with_cls , v_with_cls , block_mask=blockmask , scale=self.scale)

        clstoken = out[:,:,-num_q:,:]
        x = out[:,:,:-num_q,:]

        x = rearrange(x ,'b Nh (j i h w) Hd -> b (j h ) (i w ) ( Nh Hd) ' , Nh = self.num_heads , j = self.n_win , i = self.n_win , h = h_w , w = h_w).contiguous()

        clstoken = rearrange(clstoken , 'b  Nh  q  c -> b q (Nh  c)')

        x  = x + lepe
        x = self.wo(x)
        clstoken = self.wo(clstoken)

        return  x , clstoken

class AttentionLePE(nn.Module):
    """
    Attention with LePE (local positional encoding via depthwise conv),
    supporting an extra cls token.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(
            dim, dim, kernel_size=side_dwconv, stride=1,
            padding=side_dwconv // 2, groups=dim
        ) if side_dwconv > 0 else (lambda x: torch.zeros_like(x))

    def forward(self, x, cls_token , retAttentionMap = False):
        """
        Args:
            x: tensor of shape [B, H, W, C]
            cls_token: tensor of shape [B, 1, C]
        Returns:
            x_out: [B, H, W, C]
            cls_out: [B, 1, C]
        """
        B, H, W, C = x.shape
        x_seq = rearrange(x, 'b h w c -> b (h w) c')   # [B, N, C]
        N = x_seq.size(1)

        # concat cls token with patch tokens
        x_cat = torch.cat([cls_token, x_seq], dim=1)  # [B, 1+N, C]

        # qkv projection
        qkv = self.qkv(x_cat).reshape(B, 1 + N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, 1+N, C//heads]

        # attention
        attnMap = (q @ k.transpose(-2, -1)) * self.scale
        attn = attnMap.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1 + N, C)  # [B, 1+N, C]

        # split cls and patch outputs
        cls_out, x_out = out[:, :1, :], out[:, 1:, :]

        # add lepe only to patch tokens
        lepe = self.lepe(rearrange(x, 'b h w c -> b c h w'))
        lepe = rearrange(lepe, 'b c h w -> b (h w) c')
        x_out = x_out + lepe

        # projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        cls_out = self.proj(cls_out)
        cls_out = self.proj_drop(cls_out)

        # reshape patch tokens back to spatial form
        x_out = rearrange(x_out, 'b (h w) c -> b h w c', h=H, w=W)
        
        if retAttentionMap:
            return x_out , cls_out , attnMap
        else:
            return x_out, cls_out


class Block(nn.Module):
    def __init__(self , dim , drop_path=0 , num_heads=8, n_win=7, topk = 4 ,mlp_ratio = 3 ):
        super().__init__()
        self.attn = BiLevelRoutingAttention()
        
        self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=3, padding=1, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        pass
    def forwards(self , x , clstoken , mask): 
        resX = x + self.pos_embed(x)
        resC = clstoken

        x = self.norm1(x)
        clstoken = self.norm1(clstoken)
        x , clstoken = self.attn(x , clstoken , mask)

        x = self.drop_path(x) + resX
        clstoken = resC + clstoken
        
        resX = x
        resC = clstoken

        x = self.mlp(self.norm2(x))
        clstoken = self.mlp(self.norm2(clstoken))

        x = self.drop_path(x) + resX
        clstoken = clstoken + resC



        
        # x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
        # x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, 
        return x , clstoken


class Layer(nn.Module):
    def __init__(self,inputh,inputw , depth , topk , dim ):
        super().__init__()
        self.upsample = PatchExpand((inputh,inputw),2)
        self.blocks = nn.Sequential()
        self.clsLinear = nn.Linear(dim //2 , dim)
        for i in range(depth):
            self.blocks.append(Block())

        pass
    def forward(self , x , clstoken , skip , mask):
        x = self.upsample(x)
        clstoken = self.clsLinear(clstoken)
        x = x + skip
        x , clstoken = self.blocks(x, clstoken , mask)
        
        return x , clstoken  



    
class BasicLayer(nn.Module):
    def __init__(self,w , depth , topk , dim ):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.append(Block())

        pass
    def forward(self , x , clstoken ):
        x , clstoken = self.blocks(x, clstoken)
        return x , clstoken  

    
class BasicBlock(nn.Module):
    def __init__(self , dim , drop_path=0 , num_heads=8, n_win=7, topk = 4 ,mlp_ratio = 3  , retAttnMap = False ):
        super().__init__()
        self.attn = AttentionLePE()
        self.retAttnMap = retAttnMap
        
        self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=3, padding=1, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        pass
    def forwards(self , x , clstoken ): 
        resX = x + self.pos_embed(x)
        resC = clstoken

        x = self.norm1(x)
        clstoken = self.norm1(clstoken)
        if self.retAttnMap:
            x , clstoken , attnMap = self.attn(x , clstoken ,True )
        else:
            x , clstoken  = self.attn(x , clstoken )

        x = self.drop_path(x) + resX
        clstoken = resC + clstoken
        
        resX = x
        resC = clstoken

        x = self.mlp(self.norm2(x))
        clstoken = self.mlp(self.norm2(clstoken))

        x = self.drop_path(x) + resX
        clstoken = clstoken + resC

        if self.retAttnMap :
            return x , clstoken , attnMap
        else:
            return x , clstoken

class BiUnet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, use_checkpoints=[False, False, False, False], chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        # self.q = nn.Parameter(torch.randn(1, num_q, self.embed_dim) * 0.02)
        self.encoder =  Encoder(in_chans=in_chans, embed_dim=embed_dims,
            norm_layer=norm_layer if self.patch_norm else None)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.ModuleList()
        for i in range(3):
            self.up.append(PatchExpand((224,224) ,embed_dims[-(i+1)] , 2))
        self.up.append(nn.Identity())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward(self, x):
        x , features = self.encoder(x)
        # append cls token : x : b 1+hw , c
        for i in range(4):
            #输入： x , skip
            x = self.stage[i](x , features[i] , attentionMap)
            x = self.up(x)
        
        # b , 1 , c
        clstoken = x[:,0,:]
        clstoken = clstoken.squeeze(1)
        res = self.head(clstoken)
        return res
        







@register_model
def BiUnet_t(args):
    model = BiUnet(
        num_classes=args.nb_classes,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, False, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model
