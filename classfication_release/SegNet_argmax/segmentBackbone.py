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
from .modules import PatchEmbed , PatchMerging , DWConv2d , FeedForwardNetwork,RetNetRelPos2d,MemoryEfficientSwish,theta_shift,RotateModule


class MaskScheduler:
    """
    动态调度 soft mask / gumbel-softmax / argmax 的分区mask
    """
    def __init__(self, 
                 max_epoch: int,
                 tau_start: float = 2.0,
                 tau_end: float = 0.1,
                 soft_phase: float = 0.3,   # 前30% epoch
                 gumbel_phase: float = 0.6, # 接下来的40%
                 hard_phase: float = 0.1):  # 再接下来的10%，最后是argmax
        self.max_epoch = max_epoch
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.soft_phase = soft_phase
        self.gumbel_phase = gumbel_phase
        self.hard_phase = hard_phase

    def get_tau(self, epoch: int):
        """指数衰减温度"""
        ratio = epoch / self.max_epoch
        tau = self.tau_start * (self.tau_end / self.tau_start) ** ratio
        return tau

    def __call__(self, logits: torch.Tensor, epoch: int):
        """
        logits: [B, N, num_q]
        epoch: 当前训练的epoch
        return: mask [B, N, N] (float类型，1允许，0禁止)
        """
        tau = self.get_tau(epoch)
        B, N, num_q = logits.shape

        phase1 = int(self.soft_phase * self.max_epoch)
        phase2 = phase1 + int(self.gumbel_phase * self.max_epoch)
        phase3 = phase2 + int(self.hard_phase * self.max_epoch)
        if epoch < phase1:
            # soft mask
            p = F.softmax(logits / tau, dim=-1)       # [B, N, num_q]
            mask = torch.einsum("bik,bjk->bij", p, p) # 内积

        elif epoch < phase2:
            # gumbel-softmax soft
            p = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            mask = torch.einsum("bik,bjk->bij", p, p)

        elif epoch < phase3:
            # gumbel-softmax hard (前向one-hot，反向可微)
            p = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            mask = torch.einsum("bik,bjk->bij", p, p)

        else:
            # 最终 argmax
            region_id = logits.argmax(dim=-1) # [B, N]
            mask = (region_id.unsqueeze(-1) == region_id.unsqueeze(-2)).float()

        return mask

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

    def forward(self, x: torch.Tensor , mask , h , w):
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
        lepe = self.lepe(v[:,queriesNum :].reshape(bsz , h , w , -1)).reshape(bsz , h*w , -1)

        k *= self.scaling

        q_h = q.reshape(bsz , seq_len , self.num_heads , -1).permute(0 , 2 , 1 , 3)
        k_h = k.reshape(bsz , seq_len , self.num_heads , -1).permute(0 , 2 , 1 , 3)
        #FIXME:
        v_h = k.reshape(bsz , seq_len , self.num_heads , -1).permute(0 , 2 , 1 , 3)
        
        qk_mat = q_h @ k_h.transpose(-1, -2) #(b n l l)
        
        if mask is not None:
            # FIXME : 确认是对应元素相乘
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # 形状变为 [b, h, n, n]
            qk_mat = qk_mat * mask  #(b n l l)
        
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
    
def visualize(mask_logits , hw):
    with torch.no_grad():
        import matplotlib.pyplot as plt
        region_id = mask_logits.argmax(dim=-1)
        region_map = region_id.reshape(hw,hw)
        cmap = plt.cm.get_cmap('tab10', 8)
        plt.imshow(region_map, cmap=cmap)

def similarity(queries):
    fm = queries @ queries.transpose(-1 , -2)
    fm = fm[0]
    fm.fill_diagonal_(0)
    return fm.mean()

def classCounter(mask_logits):
    return  torch.bincount( mask_logits.argmax(dim=-1)[0], minlength=8)

class SegBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        # self.queriesNorm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mask_attention = MaskAttention(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim,subconv=False)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)
        #FIXME:
        self.mask_scheduler = MaskScheduler(max_epoch=200)
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
        # now generate mask

        
        #FIXME: 使用了full 
        mask_logits =  x @ queries.transpose(-1, -2) # b , N ,numq
        
        # region_id = mask_logits.argmax(dim=-1)
        mask = torch.ones(b,num_q+h*w , num_q+h*w,dtype=x_with_q.dtype,device=x_with_q.device)
        # B, N , N 

        # mask[:,num_q:,num_q:] = region_id.unsqueeze(-1) == region_id.unsqueeze(-2) 
        mask[:,num_q:,num_q:]  = self.mask_scheduler(mask_logits, epoch)
        
        
        # #FIXME: 
        # mask_logist = x @ queries.transpose(-1, -2) # b , N ,numq 
        # region_id = mask_logist .argmax(dim=-1) 
        # mask = torch.ones(b,num_q+h*w , num_q+h*w,dtype=x_with_q.dtype,device=x_with_q.device) # B, N , N 
        # mask[:,num_q:,num_q:] = region_id.unsqueeze(-1) == region_id.unsqueeze(-2)
        
        if self.layerscale:
            x_with_q = x_with_q + self.drop_path(self.gamma_1 * self.mask_attention(self.retention_layer_norm(x_with_q), mask , h ,w))
            x_with_q = x_with_q + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x_with_q)))
        else:
            x_with_q = x_with_q + self.drop_path(self.mask_attention(self.retention_layer_norm(x_with_q) , mask , h , w))
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
    
        
        for blk in self.blocks:
            x  , queries= blk(x , queries , epoch)
                
        if self.downsample is not None:
            x = self.downsample(x)
            queries = self.queriesLinear(queries)
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
        num_q = 8
        self.q = nn.Parameter(torch.randn(1, num_q, self.embed_dim) * 0.02)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)
        

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:  
                layer = BasicLayer(
                    embed_dim=embed_dims[i_layer],
                    out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    init_value=init_values[i_layer],
                    heads_range=heads_ranges[i_layer],
                    ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    chunkwise_recurrent=chunkwise_recurrents[i_layer],
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoints[i_layer],
                    layerscale=layerscales[i_layer],
                    layer_init_values=layer_init_values
                )
            else:
                layer = SegLayer(
                    embed_dim=embed_dims[i_layer],
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
        num_q = 8
        queries = torch.randn(b,num_q , self.embed_dim)
        queries = queries / queries.norm(dim=-1, keepdim=True)
        
        #  b c , h ,w
        x = self.patch_embed(x)
        #FIXME : N , C -> B , N , C
        # queries = self.q.weight[None , :].expand(x.shape[0] , -1,-1)
        queries = self.q.expand(b , -1,-1)
        
        firstlayer = self.layers[0]
        
        # 在这个layer内部，每个block都会做一次corss attention

        x  , queries = firstlayer(x ,queries)
        
        for layer in self.layers[1:]:
            x , queries = layer(x , queries , epoch)

        x = self.proj(x) #(b h w c)
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3) #(b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x ,epoch=200):
        x = self.forward_features(x , epoch)
        x = self.head(x)
        return x

@register_model
def VisSegNet_argmax_T(args):
    model = VisSegNet(
        num_classes= args.nb_classes,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2,8,2],
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


@register_model
def VisSegNet_argmax_S(args):
    model = VisSegNet(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()
    return model