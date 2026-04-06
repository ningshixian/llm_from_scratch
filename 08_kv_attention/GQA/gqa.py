import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    GQA = Query heads 多，Key/Value heads 少
    多个 Q head 共享同一个 K/V head

    支持:
    1. causal self-attention
    2. KV cache（增量解码）
    3. 可选 RoPE（通过 rotary_fn 注入）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_groups: int = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads        # d_k

        self.num_kv_groups = num_kv_groups if num_kv_groups is not None else num_heads  # KV head 的数量
        self.group_size = num_heads // num_kv_groups   # 每个 KV head 被多少个 Q head 共享

        # projections
        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, num_kv_groups * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, num_kv_groups * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(
        self, x, use_cache=False, rotary_fn=None, attention_mask=None
    ):
        B, T, _ = x.shape

        # 1) 线性投影
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.wv(x).view(B, T, self.num_kv_groups, self.head_dim)

        # 2) 调整维度 以便计算注意力
        q = q.transpose(1, 2)   # [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)   # [B, num_kv_groups, T, head_dim]
        v = v.transpose(1, 2)   # [B, num_kv_groups, T, head_dim]

        # 3) 可选 RoPE
        # 形如 rotary_fn(q, k) -> (q, k)
        if rotary_fn is not None:
            q, k = rotary_fn(q, k)

        # 4) 可选 KV缓存
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v

        # 5) 把 KV head 扩展到和 Q head 一样多
        # q: [B, num_heads, T, head_dim]
        # k/v: [B, num_heads, T, head_dim]
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 6) attention score
        # (B, H, T, d_k) * (B, H, d_k, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 7) causal mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)

        # 8) 外部 mask（比如 padding mask）
        if attention_mask is not None:
            scores = scores + attention_mask

        # 9) softmax
        attn = F.softmax(scores, dim=-1).type_as(q)
        attn = self.dropout(attn)

        # 10) 加权求和
        # (B, H, T, T) * (B, H, T, d_k) -> (B, H, T, d_k)
        out = torch.matmul(attn, v)

        # 11) 拼回去
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.wo(out)
        return out
    
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
