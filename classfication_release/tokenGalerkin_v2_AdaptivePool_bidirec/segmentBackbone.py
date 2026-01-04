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
    
class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
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

        self.local_representation = SwiftFormerLocalRepresentation(dim = embedding_dim ,kernel_size=3 , drop_path=0.0 , use_layer_scale=True )

        # 原文中，这里要对qkv的c做压缩，做完attention后再复原回来。。但我感觉这样损失信息太多了，暂时不采用
        # self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(embedding_dim, num_heads,batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        # self.norm4 = nn.LayerNorm(embedding_dim)
        # self.cross_attn_image_to_token = nn.MultiheadAttention(embedding_dim, num_heads,batch_first=True)
        self.cross_attn_image_to_token = simple_attn(embedding_dim , num_heads)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.mixer = nn.Linear (embedding_dim , embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe



    def forward(
        self, agent: torch.Tensor, imgs: torch.Tensor, agent_pe: torch.Tensor, imgs_pe: torch.Tensor , h , w
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        imgs = rearrange(imgs , "b ( h w ) c -> b c h w" , h = h)
        poolImgs = rearrange( self.pool(imgs) , "b c h w -> b (h w ) c")      
        local = rearrange(self.local_representation(imgs), "b c h w -> b (h w ) c")
        imgs = rearrange(imgs , " b c h w -> b ( h w ) c")
        agent = agent + poolImgs

        agent_res = agent
        norm_q = self.norm1(agent + agent_pe)
        norm_k = self.norm1(imgs + imgs_pe)
        norm_value = self.norm1(imgs)
        attn_out = self.cross_attn_token_to_image(query= norm_q, key=norm_k, value=norm_value,need_weights=False)[0]
        agent = agent_res + attn_out

        imgs_res = imgs
        norm_token = self.norm2(agent + agent_pe)
        norm_imgs = self.norm2(imgs)
        attn_out = self.cross_attn_image_to_token(x = norm_imgs , token = norm_token)
        imgs = imgs_res + attn_out
        
        imgs_res = imgs
        imgs =  imgs_res + self.mlp(self.norm3(imgs))
        
        res = imgs
        local = local * torch.sigmoid(local)
        imgs =  torch.sigmoid(imgs)
        imgs = res + self.mixer(local * imgs)

        

        return agent, imgs
class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x

class TokenAttentionLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, depth, num_heads, num_q,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample: PatchMerging=None, 
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.image_pe = PositionEmbeddingSine(self.embed_dim)
        # 定义一个包含多个卷积块的序列
        layers = []
        self.num_q = num_q
        self.q_pos = nn.Parameter(torch.randn(1,self.num_q ,self.embed_dim ))
        self.conv_layers = ConvEncoder(embed_dim , ffn_dim , kernel_size=3)


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
            queries , keys  = blk(queries  , keys,  self.q_pos ,  image_pos , h , w )
        
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


class VisSegNet(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 init_values=[1, 1, 1, 1], heads_ranges=[3, 3, 3, 3], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, use_checkpoints=[False, False, False, False], chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6,num_q = 50 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        self.num_q = num_q
        # self.q = nn.Parameter(torch.randn(1, num_q, self.embed_dim) * 0.02)
        # 删除了 0.02 ，期望不坍缩
        self.q = nn.Parameter(torch.randn(1, self.num_q, self.embed_dim))

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
                                        num_q= self.num_q,
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

# v2 : 指的是给galerkin 添加了softmax。 对图像做mlp


@register_model
def tokenGalerkin_v2_adaPool_bidirect(args):
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
        layerscales=[False, False, False, False],
        num_q = 49
    )
    model.default_cfg = _cfg()
    return model

