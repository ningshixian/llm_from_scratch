import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""支持 KV Cache 的多头自注意力（MHA），用于加速 decode 阶段

只算当前 token 的 Q
把历史 K/V 缓存起来
Attention 时直接用「历史 K/V + 当前 K/V」
"""

class KVCacheAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model  # hidden_size
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # head_dim

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # KV cache
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape

        # 1️⃣ 投影 + 多头拆分
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 2️⃣ KV cache (用于推理加速)
        if self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)  # (B, H, S_total, d_k), S_total = 历史长度 + 当前长度
            v = torch.cat([self.v_cache, v], dim=2)
        self.k_cache, self.v_cache = k, v

        # 3️⃣ attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, H, T, S_total)

        # 4️⃣ causal mask（只在 prefill 用）
        if T > 1:  # prefill 阶段一次性输入整段 prompt
            mask = torch.tril(torch.ones(T, k.shape[2]))
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # T = 1 时是 decode 阶段，每次输入一个 token，不需要 mask

        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)  # (B, H, T, d)
        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)    # (B, T, d_model)



# 🧪 Debug
torch.manual_seed(0)
attn = KVCacheAttention(d_model=64, num_heads=4)
x = torch.randn(1, 6, 64)

# # Full forward
# full_out = attn(x)
# print("Full output shape:", full_out.shape)  # (1, 6, 64)

# Incremental: prefill 4, decode 1, decode 1
out1 = attn(x[:, :4])
print("Cache K shape:", attn.k_cache.shape)  # (1, 4, 4, 16)
out2 = attn(x[:, 4:5])
out3 = attn(x[:, 5:6])
print("Cache K shape:", attn.k_cache.shape)  # (1, 4, 6, 16)
print("Incremental output shape:", out3.shape)  # (1, 1, 64)
