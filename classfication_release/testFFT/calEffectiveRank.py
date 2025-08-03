import torch
import json
from RMT import RMT_S

# 初始化模型并加载参数
model = RMT_S(None)
checkpoint = torch.load("/home/zengshimao/code/RMT/classfication_release/save/RMT-S.pth", map_location='cpu')
missingkeys, unexpect = model.load_state_dict(checkpoint['model'], strict=False)

# 模型中每个layer对应的block数量
layers_blocks = [3, 4, 18, 4]

def effective_rank(weight, threshold=0.99):
    """
    计算传入矩阵的有效秩
    参数:
        weight: torch.Tensor，要求为二维矩阵(通过.squeeze()去除末尾的单维度)
        threshold: 能量占比阈值，默认0.99
    返回:
        有效秩的比例（所需奇异值个数 / 总奇异值个数）
    """
    # weight的shape为 (out_dim, in_dim)
    # 计算SVD
    # 注意：如果矩阵不止两维，请确保只对二维部分计算（这里已通过.squeeze()确保为二维）
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    # 将奇异值转换为列表并排序（降序）
    S = sorted(S.tolist(), reverse=True)
    total_energy = sum([s**2 for s in S])
    cum_energy = 0
    for idx, s in enumerate(S, start=1):
        cum_energy += s**2
        if cum_energy / total_energy >= threshold:
            return idx / len(S)
    # 如果未达到阈值，则返回1.0
    return 1.0

# 存储所有层、block的Q、K、V有效秩结果
results = {}

for layer_idx in range(4):
    results[f"layer_{layer_idx}"] = {}
    # 每层block个数
    num_blocks = layers_blocks[layer_idx]
    for block_idx in range(num_blocks):
        results[f"layer_{layer_idx}"][f"block_{block_idx}"] = {}

        q_weight = model.layers[layer_idx].blocks[block_idx].retention.q_proj.weight.unsqueeze(-1).unsqueeze(-1)
        k_weight = model.layers[layer_idx].blocks[block_idx].retention.k_proj.weight.unsqueeze(-1).unsqueeze(-1)
        v_weight = model.layers[layer_idx].blocks[block_idx].retention.v_proj.weight.unsqueeze(-1).unsqueeze(-1)
        
        q_mat = q_weight.squeeze(-1).squeeze(-1)
        k_mat = k_weight.squeeze(-1).squeeze(-1)
        v_mat = v_weight.squeeze(-1).squeeze(-1)
        
        # 计算各自的有效秩
        q_er = effective_rank(q_mat)
        k_er = effective_rank(k_mat)
        v_er = effective_rank(v_mat)
        
        results[f"layer_{layer_idx}"][f"block_{block_idx}"]["Q"] = q_er
        results[f"layer_{layer_idx}"][f"block_{block_idx}"]["K"] = k_er
        results[f"layer_{layer_idx}"][f"block_{block_idx}"]["V"] = v_er


output_file = "effective_ranks.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"所有层所有block的Q、K、V有效秩计算完毕，结果已保存到 {output_file}")
