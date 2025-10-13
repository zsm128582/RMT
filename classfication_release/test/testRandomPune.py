import torch
import torch.nn as nn
import time
import math

# --- 1. 参数配置 (与之前相同) ---
BATCH_SIZE = 16
HEIGHT = 56
WIDTH = 56
CHANNELS = 128
NUM_HEADS = 4
NUM_CLASSES = 10
PRUNING_RATIO = 0.75
WARMUP_RUNS = 10
BENCHMARK_RUNS = 500

# --- 2. 模块实现 (有修改) ---

"""
Core of BiFormer, Bi-Level Routing Attention.

To be refactored.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index
        

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel? 
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

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
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
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
        
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v 
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad=auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
         # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            assert H%self.n_win == 0 and W%self.n_win == 0 #
        ###################################################


        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x) 

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        
        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out

    
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


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()


        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = q
        kr = k

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)

        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)

        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = self.out_proj(output)
        return output
    


class BaseAttention(nn.Module):
    # ... (这部分与之前完全相同) ...
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward_attention(self, q, k, v):
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
        output = torch.matmul(attn_probs, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(output.shape[0], -1, self.num_heads * self.head_dim)
        return self.to_out(output)

class StandardAttention(BaseAttention):
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t, qkv)
        return self.forward_attention(q, k, v)

class RG_SA(nn.Module):
    """
    Recursive-Generalization Self-Attention (RG-SA).
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(RG_SA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio) # scaled channel dimension

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM
        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(self.cr),
            nn.GELU())
        # CA
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        # CPE
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H*W
        _scale = 1

        # reduction
        _x = x.permute(0, 3,1,2).contiguous()

        if self.training:
            _time = max(int(math.log(H//4, 4)), int(math.log(W//4, 4)))
        else:
            _time = max(int(math.log(H//16, 4)), int(math.log(W//16, 4)))
            if _time < 2: _time = 2 # testing _time must equal or larger than training _time (2)

        _scale = 4 ** _time

        #


        # Recursion xT
        # 56*56的输入，降低了2次，每次降低下采样4倍？所以56*56 -》 3*3 ？？？
        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
        _x = self.norm_act(_x)

        # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
        q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
        
        # corss-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE
        # v_shape=(B, H, N', C//H)
        v = v + self.cpe(v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class PrunedAttention(BaseAttention):
    def __init__(self, dim, num_heads=8, pruning_ratio=0.5):
        super().__init__(dim, num_heads)
        self.pruning_ratio = pruning_ratio

    # === 修改点 1: 拆分剪枝逻辑为一个独立函数 ===
    def prune_kv(self, k, v):
        B, L, C = k.shape
        num_kept_tokens = int(L * (1 - self.pruning_ratio))
        
        # 简化版随机索引生成
        noise = torch.rand(L, device=k.device)
        kept_indices = torch.topk(noise, num_kept_tokens, largest=False).indices
        kept_indices = kept_indices.sort().values
        
        kept_indices_expanded = kept_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
        
        k_pruned = torch.gather(k, 1, kept_indices_expanded)
        v_pruned = torch.gather(v, 1, kept_indices_expanded)
        
        return k_pruned, v_pruned
        
    def forward(self, x, assignment_matrix):
        B, H, W, C = x.shape
        x_reshaped = x.view(B, H * W, C)

        qkv = self.to_qkv(x_reshaped).chunk(3, dim=-1)
        q, k, v = map(lambda t: t, qkv)
        
        # 将剪枝逻辑独立出来, 这样可以在外部对它计时
        k_pruned, v_pruned = self.prune_kv(k, v)
        
        return self.forward_attention(q, k_pruned, v_pruned)

# --- 3. 性能评测函数 (完全重写以正确计时) ---

def benchmark(name, model, inputs, device):
    print(f"--- Benchmarking: {name} ---")
    
    model.to(device)
    inputs_on_device = [inp.to(device) for inp in inputs]
    
    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = model(*inputs_on_device)

    # === 修改点 2: 使用 torch.cuda.Event 进行精确计时 ===
    total_time_ms = 0
    overhead_time_ms = 0

    # 定义计时事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 模拟反向传播
    output_sample = model(*inputs_on_device)
    target = torch.randn_like(output_sample)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with torch.no_grad(): # 我们只关心前向传播时间进行对比
        for _ in range(BENCHMARK_RUNS):
            # --- 测量总时间 ---
            start_event.record()
            _ = model(*inputs_on_device)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms += start_event.elapsed_time(end_event)

            # --- 如果是剪枝模型, 单独测量开销 ---
            if isinstance(model, PrunedAttention):
                # 准备K, V
                x_reshaped = inputs_on_device[0].view(BATCH_SIZE, -1, CHANNELS)
                _, k, v = model.to_qkv(x_reshaped).chunk(3, dim=-1)
                
                start_event.record()
                _, _ = model.prune_kv(k, v) # 只运行剪枝部分
                end_event.record()
                torch.cuda.synchronize()
                overhead_time_ms += start_event.elapsed_time(end_event)


    avg_total_time = total_time_ms / BENCHMARK_RUNS
    print(f"  > 平均每次前向传播时间: {avg_total_time:.4f} ms")

    if overhead_time_ms > 0:
        avg_overhead_time = overhead_time_ms / BENCHMARK_RUNS
        print(f"    - 其中, 剪枝操作 (Gather等) 开销约为: {avg_overhead_time:.4f} ms")
        print(f"    - 剪枝开销占总时间的: {avg_overhead_time/avg_total_time*100:.2f}%")
        
    return avg_total_time

# --- 4. 主程序 (有微小改动) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    image_tensor = torch.randn(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
    assignment_matrix = torch.randn(BATCH_SIZE, HEIGHT, WIDTH, NUM_CLASSES)
    
    standard_model = StandardAttention(dim=CHANNELS, num_heads=NUM_HEADS)
    pruned_model = PrunedAttention(dim=CHANNELS, num_heads=NUM_HEADS, pruning_ratio=PRUNING_RATIO)
    rmt_model = VisionRetentionChunk(embed_dim=CHANNELS , num_heads=NUM_HEADS)
    rg_model = RG_SA(dim=CHANNELS , num_heads=NUM_HEADS)
    # 总共64个window
    biformer = BiLevelRoutingAttention(dim=CHANNELS , num_heads=NUM_HEADS,topk=4)


    # 为了公平, 我们这里只对比前向传播的速度
    # 因为反向传播的对比会更复杂一些
    print("注意: 以下测试为前向传播 (Forward Pass) 的速度对比\n")
    
    standard_time = benchmark("标准ViT注意力", standard_model, [image_tensor], device)
    print()
    pruned_time = benchmark("剪枝ViT注意力", pruned_model, [image_tensor, assignment_matrix], device)
    print()
    rmt_time = benchmark("RMT注意力", rmt_model, [image_tensor], device)
    print()
    rg_time = benchmark("RG_SA注意力", rg_model, [image_tensor], device)
    print()
    # 最外面的一层，kv大约只保留了 4/64 ~ 6.25%
    biformer_time = benchmark("bigormer 注意力", biformer, [image_tensor], device)
    print()

    print("--- 总结 ---")
    if pruned_time < standard_time:
        speedup = (standard_time / pruned_time)
        reduction = (1 - pruned_time / standard_time) * 100
        print(f"剪枝版本比标准版本快 {speedup:.2f} 倍 (时间减少了 {reduction:.2f}%)。")
    else:
        slowdown = (pruned_time / standard_time)
        print(f"剪枝版本比标准版本慢 {slowdown:.2f} 倍。")
    
    print(f"\n参数配置: Batch={BATCH_SIZE}, ImgSize={HEIGHT}x{WIDTH}, Dim={CHANNELS}, PruningRatio={PRUNING_RATIO*100}%")