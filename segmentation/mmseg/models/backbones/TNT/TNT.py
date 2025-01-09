import torch
import torch.nn as nn
from .SimpleGalerkin import SelfAttention , simple_attn
from .pos_embed import get_2d_sincos_pos_embed
import math

from .edsr import make_edsr_baseline
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .SRNO import SRNO , make_coord

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...builder import BACKBONES

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True) # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]   # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MyBlock(nn.Module):

    """ TNT Block
    """
    def __init__(self, outer_dim, outer_num_heads, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def forward(self, inner_tokens, outer_tokens):
        outer_tokens[:,1:] = outer_tokens[:,1:] + inner_tokens
        # outer_tokens[:,1:] = outer_tokens[:,1:] + self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, N-1, -1)))) # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return outer_tokens
    
@BACKBONES.register_module()
class TNT(nn.Module):
    """ TNT (Transformer in Transformer) for computer vision
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, outer_dim=768, inner_dim=48,
                 depth=12, outer_num_heads=12, inner_num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, inner_stride=4, se=0):
        super().__init__()

        ###############这里改成我自己的参数####################
        inner_stride = 1 
        inner_dim = 32
        self.mask_ratio = 0.75
        in_chans = 32
        self.inner_dim = inner_dim
        self.patch_size = patch_size
        # self.img_size = img_size
        ######################################################
        
        self.num_classes = num_classes
        self.num_features = self.outer_dim = outer_dim  # num_features for consistency with other models

        self.edsr_Head = make_edsr_baseline(n_feats=in_chans,no_upsampling=True)

        self.unfold = nn.Unfold(kernel_size=patch_size , stride=patch_size)
        self.toOuterProj = nn.Conv2d(in_channels=in_chans , out_channels=outer_dim , kernel_size=patch_size , stride=patch_size)
        
        # self.num_patches  = int(img_size / patch_size) ** 2
        num_words = (self.patch_size ** 2)
        masked_num_words =  int( num_words * ( 1 -  self.mask_ratio ) )
        

        ###############add inner attention and projection############

        inner_attn = []
        self.inner_norm = nn.LayerNorm(inner_dim)
        for _ in range(2):
            inner_attn.append(simple_attn(midc=inner_dim , heads=1))
        self.inner_attns = nn.ModuleList(inner_attn)
        
        self.proj_norm1 = nn.LayerNorm(inner_dim * masked_num_words)
        self.inner_token_proj = nn.Linear(masked_num_words * inner_dim , outer_dim , bias = False)


        #############################################
        
        self.proj_norm2 = norm_layer(outer_dim)


        ########### Position embedding##############
        self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))
        # self.outer_tokens = nn.Parameter(torch.zeros(1, self.num_patches, outer_dim), requires_grad=False)
        # self.outer_pos = nn.Parameter(torch.zeros(1, self.num_patches + 1, outer_dim))

        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim),requires_grad=False) 
        pos_embed = get_2d_sincos_pos_embed(
                self.inner_pos.shape[-1], patch_size, cls_token=False)
        
        self.inner_pos.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # pos_embed = get_2d_sincos_pos_embed(
        #     self.outer_pos.shape[-1], int(img_size / patch_size) ,cls_token= True)
        
        # self.outer_pos.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(p=drop_rate)

        ############################################        
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        ##########################Attention#################
        blocks = []  
        for i in range(depth):
            blocks.append(MyBlock(
                     outer_dim=outer_dim, outer_num_heads=outer_num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))

        self.blocks = nn.ModuleList(blocks)

        ####################################################

        self.norm = norm_layer(outer_dim)

        ################FPN#################################
        # # 因为patch size 是 16 * 16 ， 经过 fpn1 以后就是上采样4倍 ， 整体来说是得到了四分之一尺度的特征
        # self.fpn1 = nn.Sequential(
        #     nn.ConvTranspose2d(outer_dim, outer_dim, kernel_size=2, stride=2),
        #     nn.SyncBatchNorm(outer_dim),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(outer_dim, outer_dim, kernel_size=2, stride=2),
        # )


        # # 八分之一尺度
        # self.fpn2 = nn.Sequential(
        #     nn.ConvTranspose2d(outer_dim, outer_dim, kernel_size=2, stride=2),
        # )

        # # 十六分之一尺度
        # self.fpn3 = nn.Identity()

        # #三十二分之一尺度
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.srno = SRNO(outer_dim,outer_dim*2,16,mod=True)

        #####################################################

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            print("load pretrained checkpoint")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            print("pretrained model is none")
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # argsort 返回的是索引值
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    


    def forward_features(self, x):
        # bchw
        B  , C , H , W = x.shape
        HP = int( H / self.patch_size)
        WP = int (W / self.patch_size)
        num_patches = HP*WP
        x = self.edsr_Head(x)
        inner_tokens = self.unfold(x).transpose(1,2).reshape(B*num_patches ,self.inner_dim , self.patch_size**2 ).transpose(1,2)+self.inner_pos

        inner_tokens, mask, id_restore = self.random_masking(inner_tokens,self.mask_ratio)
        for inattn in self.inner_attns:
            inner_tokens = inattn(self.inner_norm(inner_tokens))
        
        inner_tokens =  self.proj_norm2( self.inner_token_proj( self.proj_norm1( inner_tokens.reshape(B,num_patches , -1))))

        outer_tokens = self.proj_norm2(self.toOuterProj(x).reshape(B , -1 , num_patches).transpose(1,2))
        # outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))        
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)
        
        # outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            outer_tokens = blk(inner_tokens , outer_tokens )
        
        outer_tokens=outer_tokens[:,1:,:].permute(0,2,1).reshape(B,-1,HP,WP).contiguous()
        # h , w , 2
        h4Cood = make_coord([H/4,W/4] ,flatten=False)
        # print(h4Cood.shape)
        # b h w 2
        h4Cood = h4Cood.unsqueeze(0).repeat(B,1,1,1).cuda()

        cell4 = torch.ones(B,2)
        cell4[:, 0] *= 2 / (H/4)
        cell4[:, 1] *= 2 / (W/4)
        cell4 = cell4.cuda()

        h8Cood = make_coord([H/8,W/8],flatten=False)
        h8Cood = h8Cood.unsqueeze(0).repeat(B,1,1,1).cuda()
        cell8 = torch.ones(B,2)
        cell8[:, 0] *= 2 / (H/8)
        cell8[:, 1] *= 2 / (W/8)
        cell8 = cell8.cuda()

        fpn_features = []
        fpn_features.append(self.srno(outer_tokens,h4Cood,cell4))
        fpn_features.append(self.srno(outer_tokens,h8Cood,cell8))
        fpn_features.append(outer_tokens)
        fpn_features.append(self.fpn4(outer_tokens))
        # fpn_features = []
        # for index ,  blk in enumerate(self.blocks):
        #     outer_tokens = blk(inner_tokens, outer_tokens)
        #     if (index % 3 == 0):
        #         #BNC - > BCN
        #         fpn_features.append(outer_tokens[:,1:,:].permute(0,2,1).reshape(B,-1,HP,WP).contiguous())
        
        # ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]

        # for i in range(len(fpn_features)):
        #     fpn_features[i] = ops[i](fpn_features[i])

        return fpn_features

    def forward(self, x):
        fpn_feature=self.forward_features(x)
        
        # for i , f in enumerate(fpn_feature):
        #     print(i,"-th feature shape ",f.shape)
        # x = self.head(x)
        # if(torch.isnan(fpn_feature).any()):
        #     print("nan value detect, exit")
        #     exit(0)
        return fpn_feature


# def tnt_s_patch16_224(pretrained=False, **kwargs):
#     patch_size = 16
#     inner_stride = 4
#     outer_dim = 384
#     inner_dim = 24
#     outer_num_heads = 6
#     inner_num_heads = 4
#     del kwargs['pretrained_cfg']
#     del kwargs['pretrained_cfg_overlay']
#     del kwargs['bn_tf']
#     print(kwargs)
#     model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=12,
#                 outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
#                 inner_stride=inner_stride, **kwargs)
#     model.default_cfg = default_cfgs['tnt_s_patch16_224']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
#     return model