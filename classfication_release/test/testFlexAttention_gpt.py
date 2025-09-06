import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention, create_block_mask
)
flex_attention = torch.compile(flex_attention, dynamic=True)
torch._dynamo.config.cache_size_limit = 192
torch._dynamo.config.accumulated_cache_size_limit = 192

# ------- 模拟输入 -------
B, L, C = 4, 8192, 128        # batch, 序列长度, 隐维
H = 8                          # 多头
Nq = 16                        # 全局 queries 数
NUM_CLASSES = 64               # 分类桶数量（影响稀疏结构的粗细）

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

x = torch.randn(B, L, C, device=device, dtype=dtype)          # [B, L, C]
queries = torch.randn(B, Nq, C, device=device, dtype=dtype)   # [B, Nq, C]
class_logits = torch.randn(B, L, NUM_CLASSES, device=device, dtype=torch.float32)

# 离散化（可替换为 top-k，多标签）
classes = class_logits.argmax(dim=-1)   # [B, L], 每个token的类id 0..NUM_CLASSES-1

# ------- 构造 Q,K,V（简单线性+分头） -------
d_head = C // H
Wq = torch.nn.Linear(C, C, bias=False, device=device, dtype=dtype)
Wk = torch.nn.Linear(C, C, bias=False, device=device, dtype=dtype)
Wv = torch.nn.Linear(C, C, bias=False, device=device, dtype=dtype)

def split_heads(t):  # [B, T, C] -> [B, H, T, d_head]
    B_, T_, C_ = t.shape
    return t.view(B_, T_, H, d_head).transpose(1, 2).contiguous()

def merge_heads(t):  # [B, H, T, d_head] -> [B, T, C]
    B_, H_, T_, Dh = t.shape
    return t.transpose(1, 2).reshape(B_, T_, H_*Dh).contiguous()

# 拼接全局 queries 与普通 tokens -> [B, T, C]，T=Nq+L
tokens = torch.cat([queries, x], dim=1)
T = tokens.size(1)

Q = split_heads(Wq(tokens))          # [B, H, T, d_head]
K = split_heads(Wk(tokens))
V = split_heads(Wv(tokens))

# ------- A) 标准 SDPA（稠密mask） -------
# 规则：前 Nq 个是全局query：能看所有；其余 token i 只能看：所有全局 + 同类 tokens
# 我们构造 [B, T, T] 的布尔允许矩阵 allowed，然后转为加性 bias（0/ -inf）
allowed = torch.zeros(B, T, T, device=device, dtype=torch.bool)

# 所有位置都能看全局 keys（前Nq）
allowed[:, :, :Nq] = True

# 类内可见（对普通 tokens 区段）
for b in range(B):
    # 自身 tokens 区段 [Nq:Nq+L)
    # classes[b]: [L], 让每个 i 只能看同类 j
    cls = classes[b]  # [L]
    # 建立 [L, L] 类内相等矩阵
    same = (cls.unsqueeze(1) == cls.unsqueeze(0))  # [L, L]
    # 映射到全长索引
    allowed[b, Nq:, Nq:] = same

# 进一步：全局 queries 自身可见任何人（已经包含），并且所有人能看全局（已处理）
# 把 disallowed -> -inf， allowed -> 0.0
attn_bias = torch.zeros(B, 1, T, T, device=device, dtype=tokens.dtype)
attn_bias[~allowed.unsqueeze(1)] = float("-inf")

def sdpa_forward(Q, K, V, attn_bias):
    # F.scaled_dot_product_attention 接受 [B,H,T,D]，mask 可广播到 [B,H,T,T]
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_bias)
    return out

# 热身
_ = sdpa_forward(Q, K, V, attn_bias)

# 简单计时
def timeit(fn, iters=20):
    if device == "cuda":
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        start.record()
        for _ in range(iters):
            y = fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        return ms, y
    else:
        import time
        t0 = time.time()
        y = None
        for _ in range(iters):
            y = fn()
        ms = (time.time() - t0) * 1000 / iters
        return ms, y

sdpa_ms, y_sdpa = timeit(lambda: sdpa_forward(Q, K, V, attn_bias))
print(f"[SDPA dense mask] avg forward: {sdpa_ms:.2f} ms")

# ------- B) FlexAttention（block-sparse） -------
# 用 mask_mod 描述“允许的 (q_idx, k_idx)”；再用 create_block_mask 编译块稀疏结构
# 关键：我们需要在 mask_mod 内知道：哪些是全局（索引< Nq），哪些属于哪个类。
# 先准备每个 batch 的 token->class 映射（长度 T），前 Nq 标记为 -1（表示全局）
tok2class = torch.full((B, T), fill_value=-1, device=device, dtype=torch.int32)
tok2class[:, Nq:] = classes.to(torch.int32)

# 注意：mask_mod 的参数是标量索引（b,h,q_idx,kv_idx），需在闭包中读取 tok2class/Nq
def mask_mod(b, h, q_idx, kv_idx):
    # 任何人都能看全局 keys
    return  (kv_idx < Nq) | (q_idx < Nq) |   tok2class[b, q_idx] == tok2class[b, kv_idx]


# 可选：也可以用 score_mod 做硬 mask（把不允许的分数置 -inf）：
def score_mod(score, b, h, q_idx, kv_idx):
    allow = mask_mod(b, h, q_idx, kv_idx)
    return score if allow else torch.full_like(score, float("-inf"))

# 生成块稀疏 BlockMask（可调整 BLOCK_SIZE 以配合集群结构）
block_mask = create_block_mask(
    mask_mod, B=B, H=H, Q_LEN=T, KV_LEN=T, device=device, BLOCK_SIZE=128,_compile=True
)

def flex_forward(Q, K, V):
    # 可只传 block_mask（推荐用于加速）；若你的规则里还需要逐元素修改，可同时传 score_mod
    out = flex_attention(Q, K, V, block_mask=block_mask)
    return out

# 热身
_ = flex_forward(Q, K, V)
flex_ms, y_flex = timeit(lambda: flex_forward(Q, K, V))
print(f"[FlexAttention block-sparse] avg forward: {flex_ms:.2f} ms")

# ------- C)（可选）按类重排以获得更“块”的结构 -------
# 原序通常交替类导致 mask 分裂；把普通 tokens 按类排序，把全局放在前面，可以让 BlockMask 更密集成大块
def reorder_by_class(tokens, classes):
    # tokens: [B, L, C], classes: [B, L]
    # 返回 排序后的 tokens 与 恢复索引
    B, L, C = tokens.shape
    sorted_idx = torch.argsort(classes, dim=1)  # [B, L]
    tok_sorted = tokens.gather(dim=1, index=sorted_idx.unsqueeze(-1).expand(-1, -1, C))
    inv_idx = torch.empty_like(sorted_idx)
    inv_idx.scatter_(1, sorted_idx, torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1))
    return tok_sorted, sorted_idx, inv_idx

# 试验重排：只对普通 tokens 排序，然后再与全局拼接
x_sorted, idx_sorted, idx_inv = reorder_by_class(x, classes)
tokens2 = torch.cat([queries, x_sorted], dim=1)
Q2 = split_heads(Wq(tokens2)); K2 = split_heads(Wk(tokens2)); V2 = split_heads(Wv(tokens2))

tok2class2 = torch.full((B, Nq + L), -1, device=device, dtype=torch.int32)
tok2class2[:, Nq:] = torch.gather(classes, 1, idx_sorted).to(torch.int32)

def mask_mod2(b, h, q_idx, kv_idx):
    # return torch.where()
    return (kv_idx < Nq) | (q_idx < Nq) |   tok2class2[b, q_idx] == tok2class2[b, kv_idx]
    # return q_idx >= kv_idx


block_mask2 = create_block_mask(mask_mod2, B=B, H=H, Q_LEN=T, KV_LEN=T, device=device, BLOCK_SIZE=128,_compile = True)

def flex_forward_sorted(Q, K, V):
    return flex_attention(Q, K, V, block_mask=block_mask2)

_ = flex_forward_sorted(Q2, K2, V2)
flex_sorted_ms, y_flex_sorted = timeit(lambda: flex_forward_sorted(Q2, K2, V2))
print(f"[FlexAttention + class-sorted] avg forward: {flex_sorted_ms:.2f} ms")
