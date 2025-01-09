import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc
        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.qkv_proj.apply(init_weights)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)
        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))
        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        k = self.kln(k)
        v = self.vln(v)
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
        ret = v.permute(0, 3, 1, 2) + bias
        # 这里添加了短路连接，最坏情况下不降低性能才对啊？
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        return bias

def init_weights(m, delta = 0.01):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight, gain = 1.0e-4) + (delta * torch.diag(torch.ones(
                m.weight.size(0), m.weight.size(1), dtype=torch.float32
            )).unsqueeze(-1).unsqueeze(-1))