import torch
import time
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 必须 'cuda' for Flex
torch.manual_seed(42)

# 参数（调整 L 测试大序列）
b = 2
n = 4  # queries 数
L = 4096  # data 长度（增大测试性能）
S = n + L
d = 64  # dim
num_classes = 10

# 合成数据
logits = torch.randn(b, L, num_classes, device=device)
classes = torch.argmax(logits, dim=-1)  # [b, L]

qkv = torch.randn(b, 1, S, d, device=device)  # [b, heads=1, S, d]
q = k = v = qkv

# 标准实现（dense mask）
def standard_attn(q, k, v, classes):
    import torch.nn.functional as F
    # Additive mask [b, 1, S, S]
    mask = torch.full((b, 1, S, S), float('-inf'), device=device)
    # Queries 见所有
    mask[:, :, :n, :] = 0.0
    # Data 见 queries
    mask[:, :, n:, :n] = 0.0
    # Data 见同类 data（含自身）
    for bb in range(b):
        cls_b = classes[bb]
        data_mask = (cls_b[:, None] == cls_b[None, :]).float()  # [L, L] 向量化优化
        mask[bb, :, n:, n:] = torch.where(data_mask == 1, 0.0, float('-inf'))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out

# FlexAttention 实现（sparse block mask）
def mask_mod(b_idx, h_idx, q_idx, kv_idx):
    if q_idx < n or kv_idx < n:
        return True
    else:
        q_class = classes[b_idx, q_idx - n]
        kv_class = classes[b_idx, kv_idx - n]
        return q_class == kv_class

block_mask = create_block_mask(mask_mod, B=b, H=1, Q_LEN=S, KV_LEN=S)

def flex_attn(q, k, v, block_mask):
    out = flex_attention(q, k, v, block_mask=block_mask)
    return out

# 时间对比（跑多次取平均）
def benchmark(func, *args, num_runs=5):
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        out = func(*args)
        torch.cuda.synchronize() if device == 'cuda' else None
        times.append(time.time() - start)
    return sum(times) / num_runs

time_std = benchmark(standard_attn, q, k, v, classes)
print(f'Standard avg time: {time_std:.4f} seconds')

time_flex = benchmark(flex_attn, q, k, v, block_mask)
print(f'Flex avg time: {time_flex:.4f} seconds')

# 验证输出（应近似相等）
out_std = standard_attn(q, k, v, classes)
out_flex = flex_attn(q, k, v, block_mask)
print(f'Outputs match: {torch.allclose(out_std, out_flex, atol=1e-5)}')