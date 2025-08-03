import torch 
from RMT import RMT_S
model = RMT_S(None)

checkpoint = torch.load("/home/zengshimao/code/RMT/classfication_release/save/RMT-S.pth", map_location='cpu')

missingkeys , unexpect = model.load_state_dict(checkpoint['model'],strict=False)

mats = [
    # 将q_proj 改成 k_proj , v_proj可以计算 K, V的有效秩
    model.layers[0].blocks[0].retention.q_proj.weight.unsqueeze(-1).unsqueeze(-1)
    ]
nowrate = 1
rate=1
for i in range(1):
    for j in range(1):
        for mat in mats:
            _,now,_ = torch.linalg.svd(mat[:,:,i,j])
            now = sorted(now,reverse=True)
            total = sum([k*k for k in now])
            tmp = 0
            idx = 0
            for k in now:
                idx += 1
                tmp += k*k
                if tmp/total > 0.99:#lower maybe
                    nowrate = min(nowrate,idx/len(now))
                    # nowrate = max([u*u for u in now])/total
                    break
rate = min(rate, nowrate)
print(rate)