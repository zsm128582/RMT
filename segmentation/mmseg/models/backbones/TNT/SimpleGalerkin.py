import torch
import torch.nn as nn

from galerkin_transformer.model import SimpleAttention
from .positionEmbedding import PositionEmbeddingSine
from .ffn import FFNLayer

class SelfAttention(nn.Module):
    def __init__(self, n_feats=64 , heads = 8):
        super(SelfAttention, self).__init__()
        _attention_types = [
            "linear",
            "galerkin",
            "global",
            "causal",
            "fourier",
            "softmax",
            "integral",
            "local",
        ]
        self.n_feats = n_feats

        _norm_types = ["instance", "layer"]
        norm_type = _norm_types[1]
        attn_norm = True
        n_head = heads
        dropout = 0.1
        self.sa = SimpleAttention(
            n_head=n_head,
            d_model=n_feats,
            attention_type=_attention_types[1],
            pos_dim=-1,
            norm=attn_norm,
            norm_type=norm_type,
            dropout=0.0,
        )
        self.posEmbedding = PositionEmbeddingSine(
            num_pos_feats=n_feats // 2, normalize=True
        )
        self.dropout = nn.Dropout(dropout)

        self.layerNorm = nn.LayerNorm(n_feats)
        
        self.ffn = FFNLayer(n_feats , n_feats * 2 , normalize_before= True)
        
    def forward(self , x  ,  pos = None):
        if(len(x.shape) == 3):
            return self.tensorForward(x)
        elif(len(x.shape) == 4):
            return self.imageForward(x)
        else :
            raise("输入了啥看不懂")
        
    # x is supposed to be in shape of [ b , c , h  , w]
    def imageForward(self, x , pos = None):
        b, c, h, w = x.shape

        assert c == self.n_feats , "输入数据的维度与预期不一致！"
        # at first ,transpose x to [b , h*w , c]
        if (pos is None):
            pos = self.posEmbedding(x, None).permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        # attention
        x, _ = self.sa(query=x + pos, key=x + pos, value=x)

        x = x + self.dropout(x)

        x = self.ffn(x)

        x = x + self.dropout(x)
        # x = self.layerNorm(x)

        # transpose x back to [ b , c , h  , w]
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, -1)
        return x

    def tensorForward(self , x ):
        b, n , c = x.shape

        assert c == self.n_feats , "输入数据的维度与预期不一致！"
        # # at first ,transpose x to [b , h*w , c]
        # if (pos is None):
        #     pos = self.posEmbedding(x, None).permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            
        # x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        
        # attention
        x_, _ = self.sa(query=x , key=x , value=x)
        
        # if(torch.isnan(x_).any()):
        #     print("nan value detect after galerkin  self attention ")

        x_ = x  + self.dropout(x_)

        x_ = self.ffn(x_)

        x = x + self.dropout(x_)
        # x = self.layerNorm(x)

        # # transpose x back to [ b , c , h  , w]
        # x = x.permute(0, 2, 1).contiguous().view(b, c, h, -1)
        return x


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


class GKattn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.qkv_proj.apply(init_weights)
        #self.w = nn.init.kaiming_normal_(nn.Parameter(torch.randn(self.heads, self.headc, 3*self.headc)))
        #self.b = nn.init.kaiming_normal_(nn.Parameter(torch.randn(self.heads, 3*self.headc)))
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        #self.kln = LayerNorm((self.headc))
        #self.vln = LayerNorm((self.headc))

        #self.kin = nn.ModuleList([nn.InstanceNorm1d(self.headc) for _ in range(self.heads)])
        #self.vin = nn.ModuleList([nn.InstanceNorm1d(self.headc) for _ in range(self.heads)])

        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        #if name == 0: show_feature_map((x[:,0:10:1]),'out/edsr','feat')
        
        #x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        #qkv = x.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).reshape(B, H*W, self.heads, 3*self.headc)
        #qkv = qkv.reshape(B, H*W, self.heads, 3*self.headc)

        #x = x.permute(0, 2, 3, 1).reshape(B, H*W, self.heads, self.headc)
        #qkv = torch.einsum('bnhi,hio->bnho', x, self.w) + self.b
        
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        #ranksq = torch.linalg.matrix_rank(q.squeeze(0), tol = 1e-2)
        #ranksk = torch.linalg.matrix_rank(k.squeeze(0), tol = 1e-2)
        #ranksvo = torch.linalg.matrix_rank(v.squeeze(0), tol = 1e-2)
        #show_feature_map((k).permute(0,3,1,2).reshape(B,C,H,W)[:,0:10:1],'k',name)
        #show_feature_map((v),'v',name)
        #show_feature_map((q),'q',name)
        
        v = torch.matmul(k.transpose(-2,-1), v/(H*W) )
        #show_feature_map(v,'out/edsr/kv',name=name)
        #rankskv = torch.linalg.matrix_rank(v.squeeze(0), tol = 1e-2)
        v = torch.matmul(q, v)

        #ranksv = torch.linalg.matrix_rank(v.squeeze(0), tol = 1e-2)
        #print("rankq, rankk, rankvo rankv, rankkv:",ranksq,ranksk,ranksvo,ranksv,rankskv)

        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
        #show_feature_map((v.permute(0,3,1,2)[:,0:10:1]),'z',name)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias



class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Linear(midc , 3 * midc)
        self.o_proj1 = nn.Linear(midc, midc)
        self.o_proj2 = nn.Linear(midc, midc)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, N , C = x.shape
        bias = x

        qkv = self.qkv_proj(x).reshape(B, N, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)


        k = self.kln(k)
        v = self.vln(v)


        v = torch.matmul(k.transpose(-2,-1), v) / N

        v = torch.matmul(q, v)

        v = v.permute(0, 2, 1, 3).reshape(B, N, C)

        ret = v + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias
    
def init_weights(m, delta = 0.01):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight, gain = 1.0e-4) + (delta * torch.diag(torch.ones(
                m.weight.size(0), m.weight.size(1), dtype=torch.float32
            )).unsqueeze(-1).unsqueeze(-1))