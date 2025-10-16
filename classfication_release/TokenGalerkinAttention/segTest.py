import torch
from segmentBackbone import VisSegNet_S


model = VisSegNet_S(None)
b , h , w ,c = 4 , 224,224,3
x = torch.randn(b ,c , h , w, device='cpu')
res = model(x)
print(res.shape)




# def rearrange_by_region(x, logits):
#     """
#     x       : [B, L, C]   原始特征序列
#     logits  : [B, L, N]   每个 token 对 N 个区域的 logits
#     return  : x_perm   : [B, L, C]  重排后特征
#               perm     : [B, L]     重排索引（每行独立）
#               inv_perm : [B, L]     逆索引（恢复用）
#     """
#     B, L, N = logits.shape
#     device = logits.device

#     # 1) 区域 id
#     region_id = logits.argmax(dim=-1)            # [B, L]

#     # 2) 每个样本独立排序，同区域内保持原相对顺序
#     perm = torch.argsort(region_id, dim=-1, stable=True)   # [B, L]

#     # 3) 逆索引：inv_perm[b][perm[b]] = arange(L)
#     inv_perm = torch.empty_like(perm)
#     batch_arange = torch.arange(L, device=device).expand(B, L)   # [B, L]
#     inv_perm.scatter_(1, perm, batch_arange)                     # [B, L]

#     # 4) 重排：按行 gather
#     # 为了使用 gather，需要把特征最后一维也展开
#     x_perm = x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [B, L, C]

#     return x_perm, perm, inv_perm


# def restore_order(x_perm, inv_perm):
#     """
#     把重排后的结果恢复到原始顺序
#     x_perm   : [B, L, C]
#     inv_perm : [B, L]
#     return   : [B, L, C]
#     """
#     return x_perm.gather(1, inv_perm.unsqueeze(-1).expand(-1, -1, x_perm.size(-1)))


# B , L, C, N = 64 ,3136, 256, 8
# x = torch.randn(L, C, device='cuda')
# logits = torch.randn(L, N, device='cuda')

# # 重排
# x_perm, perm, inv_perm = rearrange_by_region(x, logits)

# sparse_attn_output = sparse_sageattn(
#         x, x, x,
#         mask_id=None,     
#         is_causal=False, 
#         tensor_layout="HND")

# # 还原
# y = restore_order(sparse_attn_output, inv_perm)


# print(y.shape)