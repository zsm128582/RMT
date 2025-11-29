import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import torch
import torch.nn.functional as F
from einops import rearrange, einsum
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from fvcore.nn import FlopCountAnalysis, flop_count_table
from typing import Tuple, Type
import time
from typing import Tuple, Union
from functools import partial
from einops import einsum
from .modules import PatchEmbed , PatchMerging , DWConv2d , FeedForwardNetwork,RetNetRelPos2d,MemoryEfficientSwish,theta_shift,RotateModule,simple_attn
from.positionEncoding import PositionEmbeddingSine
import os

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        # 原文中，这里要对qkv的c做压缩，做完attention后再复原回来。。但我感觉这样损失信息太多了，暂时不采用
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(embedding_dim, num_heads,batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        # self.cross_attn_image_to_token = nn.MultiheadAttention(embedding_dim, num_heads,batch_first=True)
        self.cross_attn_image_to_token = simple_attn(embedding_dim , num_heads)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            """"
                    self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
            """
            queries = self.self_attn(query=queries, key=queries, value=queries,need_weights=False)[0]
        else:
            q = queries + query_pe
            attn_out = self.self_attn(query=q, key=q, value=queries,need_weights=False)[0]
            queries = queries + attn_out

        
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(query=q, key=k, value=keys,need_weights=False)[0]
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        #
        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        # k = keys + key_pe # 图像
        attn_out = self.cross_attn_image_to_token(x = keys , token = q)




        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

class TokenAttentionLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample: PatchMerging=None, 
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.image_pe = PositionEmbeddingSine(self.embed_dim)
        # 定义一个包含多个卷积块的序列
        layers = []
        
        # 第一层的输入通道是 in_channels (即d)
        current_in_channels = embed_dim
        
        for i in range(2):
            # 卷积层：核大小3x3，padding=1 保持 h 和 w 不变
            conv = nn.Conv2d(
                in_channels=current_in_channels,
                out_channels=embed_dim, # 可以让所有卷积层的输出通道保持一致
                kernel_size=3,
                padding=1,
                bias=False # 通常在 BatchNorm 之后设置 bias=False
            )
            layers.append(conv)
            
            # 插入归一化层 (BatchNorm2d)
            layers.append(nn.BatchNorm2d(embed_dim))
            
            # 插入激活函数 (ReLU 是一个常用且效果良好的选择)
            layers.append(nn.ReLU(inplace=True))
            
            # 更新下一层的输入通道数
            current_in_channels = embed_dim
        
        # 使用 nn.Sequential 将所有层包装起来
        self.conv_layers = nn.Sequential(*layers)


        # build blocks
        self.blocks = nn.ModuleList([
            TwoWayAttentionBlock(self.embed_dim , num_heads , mlp_dim=ffn_dim,attention_downsample_rate=1,skip_first_layer_pe=(i == 0))
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
            self.queriesLinear = nn.Linear(embed_dim , out_dim)
        else:
            self.downsample = None
            self.queriesLinear = None
        
        
    def forward(self, x , queries_embedding , epoch):
        b, c, h ,w = x.shape
        # print("#$%"*20)
        # print(x.shape)
        # exit()
        b,n,d = queries_embedding.size()
        # b c h w 
        
        image_pos = self.image_pe(x)

        x = self.conv_layers(x)
        x = rearrange(x , 'b d h w->b (h w) d')
        image_pos = rearrange(image_pos , 'b c h w -> b  (h w) c')
        queries = queries_embedding
        keys = x


        for index , blk in enumerate(self.blocks):
            queries , keys = blk(queries  , keys,  queries_embedding ,  image_pos)
        
        queries = queries + queries_embedding
        x = keys + x
        x = rearrange(x , 'b (h w ) c -> b c h w' , h = h).contiguous()

        if self.downsample is not None:
            x = self.downsample(x)
            queries = self.queriesLinear(queries)

        return x , queries
    




def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

def get_conv2d(in_channels, 
               out_channels, 
               kernel_size, 
               stride, 
               padding, 
               dilation, 
               groups, 
               bias,
               attempt_use_lk_impl=True):
    
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding, 
                     dilation=dilation, 
                     groups=groups, 
                     bias=bias)
    

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'): # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))
   

class ResDWConv(nn.Conv2d):
    '''
    Depthwise convolution with residual connection
    '''
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
    
    def forward(self, x):
        x = x + super().forward(x)
        return x
        
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    
    def forward(self, x):
        # print(x.shape)
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()

class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )
        
    def forward(self, x):
        x = x * self.proj(x)
        return x



class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, 
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x
        

class RepConvBlock(nn.Module):

    def __init__(self, 
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 num_heads = 4,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False):
        super().__init__()
        
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        
        mlp_dim = int(dim*mlp_ratio)
        
        self.dwconv = ResDWConv(dim, kernel_size=3)
    
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )
        self.crossAttention = nn.MultiheadAttention(dim, num_heads,batch_first=True)
        self.pre_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(dim, mlp_dim,subconv=False)
        
        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        
    def forward_features(self, x  , queries):
        # x: b c h w 
        # q : b n q
        b , c , h , w = x.shape
        # print(f"enter block , {x.shape}")
        x = self.dwconv(x)
        
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        # b , h , w , c = x.shape
        x = rearrange(x , "b  c  h  w -> b (h w) c")
        queries = queries +  self.crossAttention(self.pre_layer_norm(queries) , x , x , need_weights = False)[0]
        queries = queries + self.ffn(self.final_layer_norm(queries))
        x = rearrange(x , "b (h w) c -> b c h w" ,h = h , w = w)
        
        return x , queries
    
    def forward(self, x , queries ):
        
        if self.use_checkpoint and x.requires_grad:
            x , queries = checkpoint(self.forward_features, x, queries, use_reentrant=False)
        else:
            x , queries = self.forward_features(x , queries)
        
        return x , queries
    
    
    

class MaskScheduler:
    def __init__(self, 
                 max_epoch: int,
                 tau_start: float = 2.0,
                 tau_end: float = 0.1,
                 soft_phase: float = 0.3,
                 gumbel_phase: float = 0.6,
                 hard_phase: float = 0.1):
        self.max_epoch = max_epoch
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.soft_phase = soft_phase
        self.gumbel_phase = gumbel_phase
        self.hard_phase = hard_phase

    def get_tau(self, epoch: int):
        ratio = epoch / self.max_epoch
        return self.tau_start * (self.tau_end / self.tau_start) ** ratio

    def __call__(self, logits: torch.Tensor, epoch: int):
        """
        logits: [B, N, num_q]
        return: mask, is_soft
        """
        tau = self.get_tau(epoch)
        B, N, num_q = logits.shape

        phase1 = int(self.soft_phase * self.max_epoch)
        phase2 = phase1 + int(self.gumbel_phase * self.max_epoch)
        phase3 = phase2 + int(self.hard_phase * self.max_epoch)
        
        # region_id = logits.argmax(dim=-1)
        # mask = (region_id.unsqueeze(-1) == region_id.unsqueeze(-2)).float()
        # # print(f"mask shape is {mask.shape}")
        # return mask , False
    
        if epoch < phase1:
            p = F.softmax(logits / tau, dim=-1)
            # p : N q. : N 属于 q 的概率   ： (N q * q N )
            mask = torch.einsum("bik,bjk->bij", p, p)  
            return mask, True  # soft

        elif epoch < phase2:
            p = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            mask = torch.einsum("bik,bjk->bij", p, p)
            return mask, True  # soft

        elif epoch < phase3:
            p = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            mask = torch.einsum("bik,bjk->bij", p, p)
            return mask, False  # hard

        else:
            region_id = logits.argmax(dim=-1)
            mask = (region_id.unsqueeze(-1) == region_id.unsqueeze(-2)).float()
            return mask, False  # hard

# 现在打算的是 Q到像素 ， 像素到query ， query 到query 都是互相通的
class MaskAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor , mask , h , w  , isSoftMask = True):
        '''
        x: (b, n+h*w , c)
        '''
        #FIXME:
        bsz, seq_len , _ = x.size()
        queriesNum = 8
        # 
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # b h w c
        lepe = self.lepe(v[:,queriesNum:].reshape(bsz , h , w , -1)).reshape(bsz , h*w , -1)

        k *= self.scaling

        q_h = q.reshape(bsz , seq_len , self.num_heads , -1).permute(0 , 2 , 1 , 3)
        k_h = k.reshape(bsz , seq_len , self.num_heads , -1).permute(0 , 2 , 1 , 3)
        #FIXME:WTF???????
        v_h = k.reshape(bsz , seq_len , self.num_heads , -1).permute(0 , 2 , 1 , 3) 
        
        qk_mat = q_h @ k_h.transpose(-1, -2) #(b n l l)
        
        if mask is not None:
            if isSoftMask:
                qk_mat = qk_mat + torch.log(mask.unsqueeze(1) + 1e-6)
            else :
                qk_mat = qk_mat.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, v_h) #(b n l d2)
        
        output = output.transpose(1, 2).reshape(bsz , seq_len , -1)
        
        output[:,queriesNum: , :] += lepe
        output = self.out_proj(output)
        return output


class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)

        self.rotateModule = RotateModule()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        # bhw H c
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        
        qr = self.rotateModule(q,sin,cos)
        kr = self.rotateModule(q , sin , cos) 
        # qr = theta_shift(q, sin, cos)
        # kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        
        qk_mat_w = qk_mat_w + mask_w  #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)

        #  b h w c
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)

        #  bwhc
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h =   + mask_h  #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output
    
# 前面那个尺度先用cross attention获得queries
# 后面的呢？
class VisionRetentionAll(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        
        assert h*w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        
        qr = theta_shift(q, sin, cos) #(b n h w d1)
        kr = theta_shift(k, sin, cos) #(b n h w d1)

        qr = qr.flatten(2, 3) #(b n l d1)
        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l)
        qk_mat = qk_mat + mask  #(b n l l)
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output



class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        # self.queriesNorm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        # self.retention = VisionRetentionAll(embed_dim, num_heads)
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        #FIXME:这里将ffn里面的一个dwconv关闭了
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim,subconv=False)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)
        self.crossAttention = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    def forward(
            self,
            x: torch.Tensor, 
            queries ,
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        x = x + self.pos(x)
        
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
            
        b , h , w , c = x.shape
        
        x = x.reshape( b , -1 ,c)
        queries = queries +  self.crossAttention(self.retention_layer_norm(queries) , x , x , need_weights = False)[0]
        queries = queries + self.ffn(self.final_layer_norm(queries))
        x = x.reshape( b , h , w, c)
        #在这里做cross attention 吧！随机初始化更好还是添加位置编码比较好？
        return x , queries

class SegBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.queriesNorm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mask_attention = MaskAttention(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim,subconv=False)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)
        # self.logit_temperature = nn.Parameter(torch.tensor(math.sqrt(embed_dim), dtype=torch.float32))
        # #FIXME:
        # self.mask_scheduler = MaskScheduler(max_epoch=200)
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
    # x shape : [ b , h , w ,c]
    # queries : [b , n ,c]
    def forward(
            self,
            x: torch.Tensor, 
            queries :torch.Tensor,
            epoch : int
        ):
        b , h , w, c = x.shape
        _,num_q ,_ = queries.shape
        x = x + self.pos(x)
        # b , N , c
        x = x.reshape(b,-1,c)
        x_with_q = torch.cat((queries , x) , dim=1)
        
        if self.layerscale:
            x_with_q = x_with_q + self.drop_path(self.gamma_1 * self.mask_attention(self.retention_layer_norm(x_with_q), None , h ,w , False))
            x_with_q = x_with_q + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x_with_q)))
        else:
            x_with_q = x_with_q + self.drop_path(self.mask_attention(self.retention_layer_norm(x_with_q) , None , h , w , False))
            x_with_q = x_with_q + self.drop_path(self.ffn(self.final_layer_norm(x_with_q)))

        x = x_with_q[:,num_q:,:].reshape(b,h,w,c)
        queries = x_with_q[:,:num_q,:]
        return x , queries 
    
class SegLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample: PatchMerging=None, 
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        
        # build blocks
        self.blocks = nn.ModuleList([
            SegBlock(embed_dim, num_heads, ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
            self.queriesLinear = nn.Linear(embed_dim , out_dim)
        else:
            self.downsample = None
            self.queriesLinear = None
        
        
    def forward(self, x , queries , epoch):
        b, h, w, d = x.size()
        b,n,d = queries.size()
        
        for index , blk in enumerate(self.blocks):
            x  , queries= blk(x , queries , epoch)
                
        if self.downsample is not None:
            x = self.downsample(x)
            queries = self.queriesLinear(queries)
        # print("x.requires_grad:", x.requires_grad)
        # print("queries.requires_grad:", queries.requires_grad)
        return x , queries
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging=None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint  
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, num_heads, ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
            self.queriesLinear = nn.Linear(embed_dim , out_dim)
        else:
            self.downsample = None
            self.queriesLinear = None

    def forward(self, x , queries):
        b, h, w, d = x.size()
        b,n,d = queries.size()
        
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        
        for blk in self.blocks:
            if self.use_checkpoint:
                tmp_blk = partial(blk, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
                x = checkpoint.checkpoint(tmp_blk, x)
            else:
                x , queries= blk(x , queries, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
                
        if self.downsample is not None:
            x = self.downsample(x)
            queries = self.queriesLinear(queries)
            
        return x , queries

class ConvLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, depth,kernel_size,
                 ls_init_value: float, res_scale: float,num_heads = 4,
                 mlp_ratio=4, dpr=0., norm_layer=nn.LayerNorm, use_gemm=False,
                 deploy = False, use_checkpoint=False , downsample: PatchMerging=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                RepConvBlock(
                    dim=embed_dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    ls_init_value=ls_init_value,
                    res_scale=res_scale,
                    norm_layer=LayerNorm2d,
                    drop_path=dpr[i],
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=use_checkpoint,
                )
        )
            
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
            self.queriesLinear = nn.Linear(embed_dim , out_dim)
        else:
            self.downsample = None
            self.queriesLinear = None
    
    def forward(self,  x , queries):
        b, c  , h , w = x.size()
        b,n,d = queries.size()
        
        for blk in self.blocks:
            x , queries= blk(x , queries)
        
        x = x.permute(0,2,3,1)   
        
        if self.downsample is not None:
            x = self.downsample(x)
            queries = self.queriesLinear(queries)
        
            
        return x , queries

class VisSegNet(nn.Module):

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

        # FIXME : numq = ？
        num_q = 100
        self.q = nn.Parameter(torch.randn(1, num_q, self.embed_dim) * 0.02)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)
        

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = TokenAttentionLayer(embed_dim=embed_dims[i_layer],
                                        out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                                        depth=depths[i_layer],
                                        num_heads=num_heads[i_layer],
                                        ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        norm_layer=norm_layer,
                                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                        layerscale=layerscales[i_layer],
                                        layer_init_values=layer_init_values
                                        )
            # if i_layer == 0:  
            #     #好大的核
            #     layer = ConvLayer(embed_dim= embed_dims[i_layer],
            #                       out_dim=embed_dims[i_layer+1],
            #                       depth=depths[i_layer],
            #                       kernel_size = 17,
            #                       ls_init_value = None,
            #                       res_scale = True,
            #                       num_heads=num_heads[i_layer],
            #                       mlp_ratio=mlp_ratios[i_layer],
            #                       dpr=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            #                       downsample=PatchMerging,
            #                       )
            # else:
            #     layer = SegLayer(
            #         embed_dim=embed_dims[i_layer],
            #         out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
            #         depth=depths[i_layer],
            #         num_heads=num_heads[i_layer],
            #         ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
            #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            #         norm_layer=norm_layer,
            #         downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            #         layerscale=layerscales[i_layer],
            #         layer_init_values=layer_init_values
            #     )
            self.layers.append(layer)
            
        self.proj = nn.Linear(self.num_features, projection)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

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

    def forward_features(self, x , epoch):
        b , c , h , w = x.shape
        # queries = torch.randn(b,num_q , self.embed_dim)
        # queries = queries / queries.norm(dim=-1, keepdim=True)
        
        #  输出 b c , h ,w
        x = self.patch_embed(x)
        #FIXME : N , C -> B , N , C
        # queries = self.q.weight[None , :].expand(x.shape[0] , -1,-1)
        queries = self.q.expand(b , -1, -1)
        
        # 在这个layer内部，每个block都会做一次corss attention

        for layer in self.layers:
            x , queries = layer(x , queries , epoch)
        
        #x : b c h w
        x = rearrange(x , "b c h w -> b h w c")

        x = self.proj(x) #(b h w c)
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3) #(b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)

        # b h c
        return x

    def forward(self, x ,epoch=200):
        x = self.forward_features(x , epoch)
        x = self.head(x)
        return x



@register_model
def tokenGalerkin_t(args):
    model = VisSegNet(
        num_classes= args.nb_classes,
        embed_dims=[64, 128, 256, 512],
        depths=[2,2,8,2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model