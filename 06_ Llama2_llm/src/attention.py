import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x
    
    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新塑形，合并键/值对头的数量和重复次数的维度
    )


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
        n_heads: int,
        max_seq_len: int,
        n_kv_groups: int = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_groups == 0, "n_heads must be divisible by n_kv_groups"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads        # d_k

        self.n_kv_groups = n_kv_groups if n_kv_groups is not None else n_heads  # KV head 的数量
        self.group_size = n_heads // n_kv_groups   # 每个 KV head 被多少个 Q head 共享

        # projections
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, n_kv_groups * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, n_kv_groups * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 加性 mask（贴近工程实现）
        mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # 注册为模型的缓冲区
        self.register_buffer("mask", mask)

        # self.register_buffer("cache_k", None, persistent=False)
        # self.register_buffer("cache_v", None, persistent=False)

    def forward(
        self, x, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ):
        B, T, _ = x.shape

        # 1) 线性投影
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_groups, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_groups, self.head_dim)

        # 应用旋转位置嵌入（RoPE）。
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # 对键和值进行扩展到和 Q head 一样多
        # [B, n_kv_groups, T, head_dim] → [B, n_heads, T, head_dim]
        # k = k.repeat_interleave(self.group_size, dim=1)
        k = repeat_kv(k, self.group_size)
        v = repeat_kv(v, self.group_size)

        # 将头作为批次维度处理。
        q = q.transpose(1, 2)   # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)   # [B, n_heads, T, head_dim]
        v = v.transpose(1, 2)   # [B, n_heads, T, head_dim]

        # # 可选 KV缓存
        # if use_cache:
        #     if self.cache_k is None:
        #         self.cache_k, self.cache_v = k, v
        #     else:
        #         self.cache_k = torch.cat([self.cache_k, k], dim=2)
        #         self.cache_v = torch.cat([self.cache_v, v], dim=2)
        #     k, v = self.cache_k, self.cache_v

        # 6) attention score
        # (B, H, T, d_k) * (B, H, d_k, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # mask（padding mask / causal mask）
        scores = scores + self.mask[:, :, :T, :T]

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
    
    # def reset_cache(self):
    #     self.cache_k, self.cache_v = None, None


if __name__ == "__main__":
    # 创建Attention实例
    attention_model = GroupedQueryAttention(
        dim=768,
        n_heads=8,
        max_seq_len=512,
        n_kv_groups=2)

    # 模拟输入数据
    batch_size = 1
    seq_len = 50  # 假设实际使用的序列长度为50
    dim = args.dim
    x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
    # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
    # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

    freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

    # 运行Attention模型
    output = attention_model(x, freqs_cos, freqs_sin)

    # attention出来之后的形状 依然是[batch_size, seq_len, dim]
    print("Output shape:", output.shape)