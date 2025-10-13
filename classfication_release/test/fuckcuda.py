import torch
num_patches = 16
H = 32 
W = 32
numq = 1
mask = torch.eye(num_patches)

h_w = 8

def mask_mod(b , h , q_idx , kv_idx):
    return (q_idx >= H*W) or (kv_idx >= H*W) or (mask[q_idx//(h_w * h_w) , kv_idx//(h_w * h_w) ]==1)

len = H*W+numq
res = torch.zeros(len,len)
for i in range(len):
    for j in range(len):
        res[i,j] = mask_mod(1,1,i ,j)
print(res)
