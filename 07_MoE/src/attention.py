import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: [B, T, n_kv_heads, head_dim]
    return: [B, T, n_heads, head_dim]
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=2)


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
        num_kv_heads: int = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.n_rep = num_heads // num_kv_heads   # 每个 KV head 被多少个 Q head 共享

        # projections
        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # KV cache: shape = [B, T, n_kv_heads, head_dim]
        self.register_buffer(
            "cache_k",
            torch.zeros(max_batch_size, max_seq_len, num_kv_heads, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(max_batch_size, max_seq_len, num_kv_heads, self.head_dim),
            persistent=False,
        )

    def reset_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
        rotary_fn=None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: [B, T, dim]

        start_pos:
            增量解码时，这一段 token 在整条序列中的起始位置

        use_cache:
            False -> 训练 / 全量推理
            True  -> 增量推理，写入并读取 cache

        rotary_fn:
            可选，形如 rotary_fn(q, k) -> (q, k)

        attention_mask:
            可选 mask，能 broadcast 到 [B, num_heads, T_q, T_k]
        """
        B, T, _ = x.shape

        # 1) 线性投影
        q = self.wq(x).view(B, T, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.head_dim)

        # 2) 可选 RoPE
        if rotary_fn is not None:
            q, k = rotary_fn(q, k)

        # 3) 处理 cache
        if use_cache:
            # 写入当前 step 的 K/V
            self.cache_k[:B, start_pos:start_pos + T] = k
            self.cache_v[:B, start_pos:start_pos + T] = v

            # 取出历史 + 当前
            k = self.cache_k[:B, :start_pos + T]   # [B, T_k, n_kv_heads, head_dim]
            v = self.cache_v[:B, :start_pos + T]
        else:
            # 全量模式下，不读 cache
            start_pos = 0

        # 4) 把 KV head 扩展到和 Q head 一样多
        # q: [B, T_q, num_heads, head_dim]
        # k/v: [B, T_k, num_heads, head_dim]
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 5) 调整维度到 attention 计算格式
        q = q.transpose(1, 2)   # [B, num_heads, T_q, head_dim]
        k = k.transpose(1, 2)   # [B, num_heads, T_k, head_dim]
        v = v.transpose(1, 2)   # [B, num_heads, T_k, head_dim]

        # 6) attention score
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, num_heads, T_q, T_k]

        # 8) 外部 mask（比如 padding mask）
        if attention_mask is not None:
            scores = scores + attention_mask

        # 9) softmax
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        attn = self.dropout(attn)

        # 10) 加权求和
        out = torch.matmul(attn, v)   # [B, num_heads, T_q, head_dim]

        # 11) 拼回去
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        out = self.wo(out)
        return out