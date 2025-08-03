import sys
import os
import torch.nn.functional as F

# 获取当前文件（sparseTest.py）的绝对路径，然后找到主目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import torch
import SparseAtten

# from cudaGatherFunction import GatherFunction


from einops import rearrange
from SpaseNet import TopkRouting , KVGather

def cuda_atten(qpatch , kpatch , vpatch , scale , num_selected , r_idx):
    attn = SparseAtten.sparse_attention_wrap(qpatch, kpatch,r_idx ,  scale )
    return attn



def pytorch_attn(qpatch , kpatch , vpatch , scale , num_selected , r_idx):
    B, heads, region, w2, c = kpatch.shape
    # b h r k w2 c
    gatherK = torch.gather(
        kpatch.view(B,heads,region,1,w2,c).expand(-1,-1,-1,region,-1,-1),
        dim=2,
        index=r_idx.view(B,heads,region,num_selected,1,1).expand(-1,-1,-1,-1,w2,c)
        )
    gatherK = rearrange(gatherK,'b h r k w2 c -> (b r) h c (k w2)')
    qpatch = rearrange(qpatch,'b heads r w2 c  -> (b r) heads w2 c')
    # (b r) h w2 (k w2)
    attn_fine = (qpatch  @ gatherK)
    attn_fine = rearrange(attn_fine , '(b r) heads w1 (k w2) -> b heads r w1 k w2' , r = region , k = num_selected)
    return attn_fine

def weightingforwardTest():
    B = 1
    H = 4
    REGION = 49
    W2 = 64
    C = 64
    scale = 1
    num_selected = 8
    qpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    kpatch= torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    vpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    comQ = qpatch.mean(dim = -2).contiguous()
    comK = kpatch.mean(dim = -2).contiguous()
    comV = vpatch.mean(dim = -2).contiguous()

    router = TopkRouting(qk_dim=C,
                                qk_scale=scale,
                                topk=num_selected,
                                diff_routing=False,
                                param_routing=False)
    r_weight, r_idx = router(comQ, comK) # both are (n, p^2, topk) tensors

    qclone = qpatch.clone()
    kclone = kpatch.clone()
    idx_clone = r_idx.clone()
    cscale = scale
    cselect = num_selected

    attn = cuda_atten(qclone, kclone , vpatch , cscale, cselect, idx_clone.contiguous())


    attn = attn.flatten(-2).softmax(-1)
    cuda_attn = attn.reshape(B , H , REGION , W2 , num_selected , W2)

    cuda_result = SparseAtten.weighting_forward_warp(cuda_attn, vpatch ,r_idx ,  scale )

        # b h r k w2 c
    gatherV = torch.gather(
        vpatch.view(B,H,REGION,1,W2,C).expand(-1,-1,-1,REGION,-1,-1),
        dim=2,
        index=r_idx.view(B,H,REGION,num_selected,1,1).expand(-1,-1,-1,-1,W2,C)
        )
    # print(gatherV)
    gatherV = rearrange(gatherV , 'b h r k w2 c -> b h r (k w2) c')
    #  b h r w2 k w2 - > b h r w2 kw2

    # # print(attn.shape)
    # # print(gatherV.shape)

    out_fine = attn @ gatherV
    # cuda_result2 = SparseAtten.weighting_forward_warp(cuda_attn, vpatch ,r_idx ,  scale )
    assert torch.allclose(cuda_result, out_fine, atol=1e-5), "CUDA and PyTorch results do not match!"
    # print(attn)
    # print(vpatch)
    # cuda_attn = attn.reshape(B , H , REGION , W2 , num_selected , W2)
    # pytorch_atten_result = pytorch_attn(qpatch , kpatch , vpatch , scale , num_selected , r_idx)

    # # attn = cuda_atten(qclone, kclone , vpatch , cscale, cselect, idx_clone.contiguous())
    # assert torch.allclose(attn, pytorch_atten_result, atol=1e-5), "CUDA and PyTorch results do not match!"
    print("weighting forward match")
   

def attnforwardTest():
    B = 16
    H = 4
    REGION = 49
    W2 = 64
    C = 64
    scale = 1
    num_selected = 12
    qpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    kpatch= torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    vpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    comQ = qpatch.mean(dim = -2).contiguous()
    comK = kpatch.mean(dim = -2).contiguous()
    comV = vpatch.mean(dim = -2).contiguous()

    router = TopkRouting(qk_dim=C,
                                qk_scale=scale,
                                topk=num_selected,
                                diff_routing=False,
                                param_routing=False)
    r_weight, r_idx = router(comQ, comK) # both are (n, p^2, topk) tensors

    qclone = qpatch.clone()
    kclone = kpatch.clone()
    idx_clone = r_idx.clone()
    cscale = scale
    cselect = num_selected

    attn = cuda_atten(qclone, kclone , vpatch , cscale, cselect, idx_clone.contiguous())


    # attn = attn.flatten(-2).softmax(-1)
    # print(attn)
    # print(vpatch)
    # cuda_attn = attn.reshape(B , H , REGION , W2 , num_selected , W2)
    pytorch_atten_result = pytorch_attn(qpatch , kpatch , vpatch , scale , num_selected , r_idx)

    attn = cuda_atten(qclone, kclone , vpatch , cscale, cselect, idx_clone.contiguous())
    assert torch.allclose(attn, pytorch_atten_result, atol=1e-5), "CUDA and PyTorch results do not match!"
    print("attn forward match")

    
    # # cuda_attn = F.softmax(cuda_attn ,dim=(-2,-1) )

    # cuda_result = SparseAtten.weighting_forward_warp(cuda_attn, vpatch ,r_idx ,  scale )

    # # b h r k w2 c
    # gatherV = torch.gather(
    #     vpatch.view(B,H,REGION,1,W2,C).expand(-1,-1,-1,REGION,-1,-1),
    #     dim=2,
    #     index=r_idx.view(B,H,REGION,num_selected,1,1).expand(-1,-1,-1,-1,W2,C)
    #     )
    # # print(gatherV)
    # gatherV = rearrange(gatherV , 'b h r k w2 c -> b h r (k w2) c')
    # #  b h r w2 k w2 - > b h r w2 kw2

    # # print(attn.shape)
    # # print(gatherV.shape)

    # out_fine = attn @ gatherV
    # # print(out_fine)
    # # print(out_fine.shape)


    # # print("pytorch:,",pytorch_attn.shape)
    # assert torch.allclose(cuda_result, out_fine, atol=1e-5), "CUDA and PyTorch results do not match!"
    # print("forward match")

def checkAttenBackward():
    B = 1
    H = 1
    REGION = 16
    W2 = 16
    C = 16
    scale = 1
    num_selected = 4
    query = torch.randn(B, H, REGION, W2, C, dtype=torch.float64, device='cuda', requires_grad=True)
    key = torch.randn(B, H, REGION, W2, C, dtype=torch.float64, device='cuda', requires_grad=True)
    comQ = query.mean(dim = -2).contiguous()
    comK = key.mean(dim = -2).contiguous()
    router = TopkRouting(qk_dim=C,
                                qk_scale=scale,
                                topk=num_selected,
                                diff_routing=False,
                                param_routing=False)
    r_weight, r_idx = router(comQ, comK) # both are (n, p^2, topk) tensors


    # idx = torch.randint(B , H , REGION , num_selected, device='cuda', dtype=torch.int64)
    # qpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    # kpatch= torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    # vpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    # comQ = qpatch.mean(dim = -2).contiguous()
    # comK = kpatch.mean(dim = -2).contiguous()
    # comV = vpatch.mean(dim = -2).contiguous()

    # router = TopkRouting(qk_dim=C,
    #                             qk_scale=scale,
    #                             topk=num_selected,
    #                             diff_routing=False,
    #                             param_routing=False)
    # r_weight, r_idx = router(comQ, comK) # both are (n, p^2, topk) tensors
    # # 使用 gradcheck 检查
    from torch.autograd import gradcheck
    from NewSparse import gatherAttention
    test = gradcheck(gatherAttention.apply, (query , key , r_idx , scale), eps=1e-6, atol=1e-4)
    print("attn backward Gradcheck passed?", test)
    # # 构造测试数据
    # # B, H, R, W2, C = 4, 4, 49, 64, 64
    # B, H, R, W2, C = 1, 1, 49, 64, 64
    # num_selected = 12
    # # 使用双精度，并确保需要梯度
    # input_tensor = torch.randn(B, H, R, W2, C, dtype=torch.float64, device='cuda', requires_grad=True)
    # # 构造一个固定的 r_idx ，无需梯度
    # r_idx = torch.randint(0, R, (B, H, R, num_selected), device='cuda', dtype=torch.int64)
    # gatherk = GatherFunction.apply(input_tensor, r_idx)
    # print(gatherk.shape)
    # from torch.autograd import gradcheck
    # # 使用 gradcheck 检查
    # test = gradcheck(GatherFunction.apply, (input_tensor, r_idx), eps=1e-6, atol=1e-4)
    # print("Gradcheck passed?", test)

def checkWeightingBackward():
    B = 1
    H = 2
    REGION = 4
    W2 = 16
    C = 16
    scale = 1
    num_selected = 2
    query = torch.randn(B, H, REGION, W2, C, dtype=torch.float64, device='cuda', requires_grad=False)
    key = torch.randn(B, H, REGION, W2, C, dtype=torch.float64, device='cuda', requires_grad=False)


    value = torch.randn(B, H, REGION, W2, C, dtype=torch.float64, device='cuda', requires_grad=True)

    comQ = query.mean(dim = -2).contiguous()
    comK = key.mean(dim = -2).contiguous()

    router = TopkRouting(qk_dim=C,
                                qk_scale=scale,
                                topk=num_selected,
                                diff_routing=False,
                                param_routing=False)

    r_weight, r_idx = router(comQ, comK) # both are (n, p^2, topk) tensors
    atten = torch.randn(B, H, REGION, W2,  num_selected , W2, dtype=torch.float64, device='cuda', requires_grad=True)
    atten = atten.flatten(-2).softmax(-1)
    atten = atten.reshape(B , H , REGION , W2 , num_selected , W2)

    # idx = torch.randint(B , H , REGION , num_selected, device='cuda', dtype=torch.int64)
    # qpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    # kpatch= torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    # vpatch = torch.randn(B, H, REGION, W2, C).cuda().contiguous()
    # comQ = qpatch.mean(dim = -2).contiguous()
    # comK = kpatch.mean(dim = -2).contiguous()
    # comV = vpatch.mean(dim = -2).contiguous()

    # router = TopkRouting(qk_dim=C,
    #                             qk_scale=scale,
    #                             topk=num_selected,
    #                             diff_routing=False,
    #                             param_routing=False)
    # r_weight, r_idx = router(comQ, comK) # both are (n, p^2, topk) tensors
    # # 使用 gradcheck 检查
    from torch.autograd import gradcheck
    from NewSparse import AttentionWeighting
    test = gradcheck(AttentionWeighting.apply, (atten , value , r_idx , scale), eps=1e-6, atol=1e-4)
    print(" weighting Gradcheck passed?", test)

if __name__ == "__main__":
    checkWeightingBackward()
    attnforwardTest()
    weightingforwardTest()
    checkAttenBackward()
    # forwardTest()
    # checkBackward()
    # testGatherFunction()