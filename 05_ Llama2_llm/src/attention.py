import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb, _precompute_freqs_cis

"""GQA (Grouped Query Attention) 

核心思想是将查询（Q）头的数量设置为键/值（K/V）头的倍数，从而实现更高效的注意力计算。具体来说：
Q：多头（num_heads）
K/V：少头（num_kv_heads）
K/V 会被 repeat 到 Q 的头数

支持
- RoPE：作用在 Q 和 K 上（非常关键！）
- KV cache
"""


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: (B, T, n_kv_heads, head_dim)
    -> (B, T, n_heads, head_dim)
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, max_seq_len=512):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = d_model // num_heads        # d_k
        self.repetitions = num_heads // num_kv_heads   # 每个 KV head 被多少个 Q head 共享

        # Q 全头
        self.wq = nn.Linear(d_model, num_heads * self.dim_head, bias=False)
        # K/V 少头
        self.wk = nn.Linear(d_model, num_kv_heads * self.dim_head, bias=False)
        self.wv = nn.Linear(d_model, num_kv_heads * self.dim_head, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(0.1)

        # causal mask（注册 buffer，一次构建）
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask, persistent=False)
        # KV cache（可选）
        self.cache_k = None
        self.cache_v = None
    
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

    def forward(self, x, freqs_cos=None, freqs_sin=None, use_cache=False):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.num_heads, self.dim_head)    # B, T, H, d_k
        k = self.wk(x).view(B, T, self.num_kv_heads, self.dim_head) # B, T, H_kv, d_k
        v = self.wv(x).view(B, T, self.num_kv_heads, self.dim_head) # B, T, H_kv, d_k

        # 应用RoPE（只作用 Q/K）
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # KV Cache （针对自回归推理）
        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=1)
                v = torch.cat([self.cache_v, v], dim=1)
            self.cache_k, self.cache_v = k, v

        # GQA：扩展 KV 到 Q 的头数
        # [B, n_kv_groups, T, head_dim] → [B, n_heads, T, head_dim]
        k = repeat_kv(k, self.repetitions)
        v = repeat_kv(v, self.repetitions)

        # 将头作为批次维度处理 -> (B, H, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力（与 MHA 相同）
        scores = (q @ k.transpose(-2, -1)) / (self.dim_head ** 0.5)    # (B, H, T, d_k) * (B, H, d_k, T) -> (B, H, T, T)
        # causal mask（裁剪到当前 T）
        scores.masked_fill_(self.causal_mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, H, T, d_k)

        # 合并 heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
    

if __name__ == "__main__":
    # 配置参数
    dim = 512
    n_heads = 8
    n_kv_groups = 2  # 8个Q头，2个KV头，每组4个Q共享1组KV
    seq_len = 64
    batch_size = 2

    # 初始化模型
    model = GroupedQueryAttention(
        d_model=dim,
        num_heads=n_heads,
        num_kv_heads=n_kv_groups
    )

    # 模拟输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 模拟 RoPE 频率
    # head_dim = dim // n_heads = 64
    f_cos, f_sin = _precompute_freqs_cis(dim // n_heads, seq_len)

    # 前向传播
    output = model(x, f_cos, f_sin)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证输出维度
    assert output.shape == (batch_size, seq_len, dim), "维度校验失败！"
    print("GQA 实现验证成功！")