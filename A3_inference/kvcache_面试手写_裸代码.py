"""
KV Cache 面试手写题 · 裸代码速查
================================================
覆盖高频考点：
  1. MHA + KV Cache          （必考：prefill / decode 两阶段）
  2. 增量生成 generate        （必考：缓存怎么用）
  3. 预分配 KV Cache          （进阶：避免 torch.cat 频繁拷贝）
  4. GQA + KV Cache           （进阶：KV 头数 < Q 头数，缓存更小）
  5. MLA + KV Cache           （进阶：只缓存低秩 latent，省显存）
  6. 显存计算公式             （口答题）

约定：B=batch, T=当前输入长度, S=缓存总长度(含当前), H=q头数, G=kv头数, d=head_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1️⃣ MHA + KV Cache（最基础版本，先背这个）
# ============================================================
class MHAWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # 缓存不进 state_dict，不参与训练
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape

        # ① 投影 + 拆头: (B, T, d) -> (B, H, T, d_k)
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # ② 拼缓存（decode 核心：只算新 token 的 Q，K/V 用 历史+当前）
        if self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)   # (B, H, S, d_k)
            v = torch.cat([self.v_cache, v], dim=2)
        self.k_cache, self.v_cache = k, v

        # ③ 注意力分数: (B, H, T, S)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # ④ causal mask：只有 prefill (T>1) 需要；decode (T=1) 天然可见全部历史
        if T > 1:
            S = k.shape[2]
            mask = torch.tril(torch.ones(T, S, dtype=torch.bool, device=x.device))
            # 通用写法（应对 prefill 之后还有 chunked prefill）：
            # mask = q_pos[:, None] >= k_pos[None, :]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1) @ v          # (B, H, T, d_k)
        out = attn.transpose(1, 2).contiguous().view(B, T, -1)  # 合并 heads
        return self.W_o(out)

    def reset_cache(self):
        self.k_cache = self.v_cache = None


# ============================================================
# 2️⃣ 增量生成：prefill 一次 + decode 逐 token
# ============================================================
@torch.no_grad()
def generate(model, x, max_new_tokens):
    """x: (B, T_prompt)。prefill 填满缓存，之后每次只喂 1 个 token。"""
    model.reset_cache()
    out = model(x)                      # prefill：整段 prompt 一次算完，缓存 K/V
    for _ in range(max_new_tokens):
        next_tok = out[:, -1:].argmax(-1, keepdim=True)   # (B, 1)
        x = torch.cat([x, next_tok], dim=1)
        out = model(next_tok)           # decode：只喂新 token，K/V 从缓存取
    return x


# ============================================================
# 3️⃣ 进阶：预分配 KV Cache（vLLM / HF 思路，面试加分项）
#    torch.cat 每步都拷贝 O(S)，预分配后写入 O(1)
# ============================================================
class MHAWithPreallocCache(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=4096):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # 提前开好最大长度的显存，用 pos 指针记录写到哪了
        self.register_buffer("k_cache", torch.zeros(1, num_heads, max_seq_len, self.d_k), persistent=False)
        self.register_buffer("v_cache", torch.zeros(1, num_heads, max_seq_len, self.d_k), persistent=False)
        self.pos = 0   # 当前已缓存的 token 数

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 写入缓存槽位，再切片出有效部分
        self.k_cache[:B, :, self.pos:self.pos + T] = k
        self.v_cache[:B, :, self.pos:self.pos + T] = v
        k_all = self.k_cache[:B, :, :self.pos + T]   # (B, H, S, d_k)
        v_all = self.v_cache[:B, :, :self.pos + T]
        self.pos += T

        scores = q @ k_all.transpose(-2, -1) / math.sqrt(self.d_k)
        if T > 1:  # prefill
            q_pos = torch.arange(self.pos - T, self.pos, device=x.device)
            k_pos = torch.arange(self.pos, device=x.device)
            mask = q_pos[:, None] < k_pos[None, :]   # True = 要屏蔽
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1) @ v_all
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, T, -1))

    def reset_cache(self):
        self.pos = 0


# ============================================================
# 4️⃣ GQA + KV Cache：K/V 只有 G 个头，缓存量 = MHA 的 G/H
#    MHA: G=H   GQA: 1<G<H   MQA: G=1
# ============================================================
class GQAWithKVCache(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)   # ⭐ KV 投影维度更小
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer("k_cache", None, persistent=False)  # (B, G, S, d_k)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        if self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)
        self.k_cache, self.v_cache = k, v

        # ⭐ KV 头广播到 Q 头数：repeat_interleave（[K1,K2] -> [K1,K1,K2,K2]）
        k = k.repeat_interleave(self.group_size, dim=1)   # (B, H, S, d_k)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if T > 1:
            S = k.shape[2]
            mask = torch.tril(torch.ones(T, S, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1) @ v
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, T, -1))


# ============================================================
# 5️⃣ MLA + KV Cache：不缓存 K/V，只缓存低秩 latent c (latent_dim << d_model)
#    DeepSeek-V2/V3 核心：显存再降一个量级
# ============================================================
class MLAWithKVCache(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.latent_dim = latent_dim
        self.W_q = nn.Linear(d_model, d_model)
        self.W_dkv = nn.Linear(d_model, latent_dim)       # ⭐ 下投影：x -> 低秩 c
        self.W_uk = nn.Linear(latent_dim, d_model)        # 上投影：c -> K
        self.W_uv = nn.Linear(latent_dim, d_model)        # 上投影：c -> V
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer("c_cache", None, persistent=False)  # (B, S, latent_dim) ⭐ 只有一份、无头维

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        c_new = self.W_dkv(x)                             # (B, T, latent_dim)

        if self.c_cache is not None:
            c = torch.cat([self.c_cache, c_new], dim=1)
        else:
            c = c_new
        self.c_cache = c                                  # ⭐ 缓存 latent，不是 K/V

        k = self.W_uk(c).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        v = self.W_uv(c).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if T > 1:
            S = k.shape[2]
            mask = torch.tril(torch.ones(T, S, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1) @ v
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, T, -1))


# ============================================================
# 6️⃣ 口答：KV Cache 显存公式（每层每 token 的字节数）
# ============================================================
# MHA : 2(K,V) × H × d_k × dtype            = 2 × d_model × bytes
# GQA : 2 × G × d_k × bytes                 = MHA × (G/H)
# MLA : latent_dim × bytes                  （K/V 共用一个 latent，只有 1 份）
# 总量 = 每层 × n_layers × B × S
# 例：LLaMA-7B (32层, d_model=4096, fp16) 每 token ≈ 32 × 2 × 4096 × 2B = 0.5 MB
#     → 1M token 上下文 ≈ 500 GB → 必须 GQA/MLA/量化/PagedAttention


# ============================================================
# ✅ 自测：增量结果 == 全量结果
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, H, G, T = 64, 4, 2, 6
    x = torch.randn(1, T, d_model)

    for cls, kw in [(MHAWithKVCache, dict(d_model=d_model, num_heads=H)),
                    (GQAWithKVCache, dict(d_model=d_model, num_heads=H, num_kv_heads=G)),
                    (MLAWithKVCache, dict(d_model=d_model, num_heads=H, latent_dim=16))]:
        torch.manual_seed(0)
        m1 = cls(**kw).eval()
        full = m1(x)[:, -1]                                # 全量前向，取最后位置
        torch.manual_seed(0)
        m2 = cls(**kw).eval()
        m2(x[:, :4])                                       # prefill 前 4 个
        inc = m2(x[:, 4:5])                                # decode
        inc = m2(x[:, 5:6])                                # decode
        assert torch.allclose(full, inc[:, -1], atol=1e-5), cls.__name__
        print(f"{cls.__name__}: 增量 == 全量 ✅")

    # 预分配版本
    torch.manual_seed(0)
    m1 = MHAWithPreallocCache(d_model, H, max_seq_len=16).eval()
    full = m1(x)[:, -1]
    torch.manual_seed(0)
    m2 = MHAWithPreallocCache(d_model, H, max_seq_len=16).eval()
    m2(x[:, :4]); m2(x[:, 4:5]); inc = m2(x[:, 5:6])
    assert torch.allclose(full, inc[:, -1], atol=1e-5)
    print("MHAWithPreallocCache: 增量 == 全量 ✅")
