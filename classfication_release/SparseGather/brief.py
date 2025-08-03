# 我写了一份两层稀疏注意力的代码，其主要思想是：先使用先将图像压缩得到粗糙表示，利用粗糙表示计算原图像块与块之间的相似度，随后计算精细注意力时，仅在相似的块之间做精细注意力。
# 首先对于输入 b c h w ， 得到q，k，v ,分头
# 随后分成patch*patch的小块，shape变成： b heads ， num_region , w^2 , dim_heads
qpatch = rearrange(q,'b heads (j h) (i w) c  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads)
kpatch = rearrange(k,'b heads (j h) (i w) c  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads)
vpatch = rearrange(v,'b heads (j h) (i w) c  -> b heads (j i) (h w) c',h=self.patch_size,w=self.patch_size , heads = self.heads)
# 使用平均得到粗糙注意力：
comQ = qpatch.mean(dim = -2)
comK = kpatch.mean(dim = -2)
comV = vpatch.mean(dim = -2)
# 随后计算块与块之间的注意力,并取topk（每个块只与前k个块相关联）

attn_logit = (comQ*self.scale) @ comK.transpose(-2, -1) # (n, w^2, w^2)
topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, w^2, k), (n, w^2, k)
r_weight = self.routing_act(topk_attn_logit) # (n, w^2, k)

#随后使用gather，将每个patch相关联前k个patch提取出来， 对于 gatherV也是一样的处理方法 ，得到的shape为：b heads num_region k w^2 c
gatherK = torch.gather(
    kpatch.view(B,heads,region,1,w2,c).expand(-1,-1,-1,region,-1,-1),
    dim=3,
    index=topk_index.view(B,heads,region,topk,1,1).expand(-1,-1,-1,-1,w2,c)
    )

# 最后按照如下方法计算精细注意力
gatherK = rearrange(gatherK,'b h r k w2 c -> (b r) h c (k w2)')
gatherV = rearrange(gatherV,'b h r k w2 c -> (b r) h (k w2) c')  
# br h w2 c
qpatch = rearrange(qpatch,'b heads r w2 c  -> (b r) heads w2 c')
attn_fine = ((qpatch * self.scale) @ gatherK).softmax(-1)
# br h w2 c
out_fine = attn_fine @ gatherV
out_fine = rearrange(out_fine,'(b r) h w2 c -> b h (r w2) c',r=num_regions)
#