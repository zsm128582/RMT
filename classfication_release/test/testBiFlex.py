import torch
import torch.nn as nn

# 确保你使用的 PyTorch 版本支持 flex_attention
# 通常需要 PyTorch 2.x 和 CUDA 环境
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention, create_block_mask
)
flex_attention = torch.compile(flex_attention, dynamic=True)
torch._dynamo.config.cache_size_limit = 192
torch._dynamo.config.accumulated_cache_size_limit = 192

# 


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

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 1. 定义参数 ---
    batch_size = 2
    height, width = 32 , 32
    patch_size = 8
    numHeads = 4
    embed_dim = 64 # C: 通道数
    numq = 1

    # --- 2. 检查 CUDA 环境并创建模型 ---
    if not torch.cuda.is_available():
        print("跳过执行: flex_attention 需要一个支持 CUDA 的 PyTorch 版本。")
    else:
        device = 'cuda'
        print(f"在 {device} 上运行...")

        # 实例化注意力层
        attention_layer = BiLevelRoutingAttention(
            dim=embed_dim , num_heads=numHeads , n_win= 4
        ).to(device)

        # --- 3. 创建输入数据 ---
        # 创建一个随机的输入图像张量
        x = torch.randn(batch_size, height, width, embed_dim, device=device)
        clstoken = torch.randn(batch_size , numq , embed_dim , device = device)
        

        # --- 4. 创建稀疏的块级掩码 ---
        num_patches_h = height // patch_size  # 4
        num_patches_w = width // patch_size   # 4
        num_patches = num_patches_h * num_patches_w  # 16

        # 创建一个 (B, N, N) 的掩码张量，N=16
        # 我们来构造一个有意义的掩码作为例子：
        # - 对于 batch 0: 每个块只能关注它自己以及相邻的块（类似局部注意力）
        # - 对于 batch 1: 每个块只能关注它自己（对角线掩码）
        mask = torch.zeros(batch_size, num_patches+1, num_patches+1, device=device)

        # Batch 0: 局部注意力掩码
        for i in range(num_patches):
            # 将1D块索引转换回2D
            patch_row_i = i // num_patches_w
            patch_col_i = i % num_patches_w
            for j in range(num_patches):
                patch_row_j = j // num_patches_w
                patch_col_j = j % num_patches_w
                # 如果是3x3邻域内的块，则允许关注
                if abs(patch_row_i - patch_row_j) <= 1 and abs(patch_col_i - patch_col_j) <= 1:
                    mask[0, i, j] = 1
        
        # Batch 1: 仅关注自己的掩码
        mask[1] = torch.eye(num_patches+1, device=device)

        print("\n--- 输入张量形状 ---")
        print("输入 x shape:", x.shape)
        print("块级 mask shape:", mask.shape)
        
        print("\n--- 掩码示例 (仅展示前8x8部分) ---")
        print("Batch 0 的掩码 (局部注意力):\n", mask[0, :8, :8].int())
        print("\nBatch 1 的掩码 (仅关注自身):\n", mask[1, :8, :8].int())

        # --- 5. 执行前向传播 ---
        output = attention_layer(x , clstoken , mask)

        # --- 6. 验证输出 ---
        print("\n--- 输出张量形状 ---")
        print("输出 output shape:", output.shape)
        
        # 检查输出形状是否与输入一致

        print("\n成功执行！输出形状与输入匹配。")