import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""GQA (Grouped Query Attention) 
目的：减少 KV Cache 缓存占用

假设配置：$H = 32$，$H_{kv} = 8$，$d_k = 128$。
- 分组大小 $G = 32 / 8 = 4$。
- 32 个 Query 头被分成 8 个组，每组 4 个头。
- 第 0 组：Query 头 $[0, 1, 2, 3]$ 一起使用 KV 头 0。
- 第 1 组：Query 头 $[4, 5, 6, 7]$ 一起使用 KV 头 1。
依此类推……

支持
- RoPE：作用在 Q 和 K 上（非常关键！）
- KV cache
"""

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len, qkv_bias=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads# Query 头数，记为 H_q
        self.num_kv_heads = num_kv_heads  # Key/Value 头数，记为 H_kv
        self.d_k = d_model // num_heads  # 每个头的维度，记为 head_dim
        self.repetitions = num_heads // num_kv_heads   # K/V 会被 repeat 到 Q 的头数

        # Q 全头
        self.wq = nn.Linear(d_model, num_heads * self.d_k, bias=qkv_bias)
        # K/V 少头
        self.wk = nn.Linear(d_model, num_kv_heads * self.d_k, bias=qkv_bias)
        self.wv = nn.Linear(d_model, num_kv_heads * self.d_k, bias=qkv_bias)
        self.wo = nn.Linear(d_model, d_model)

        # # 注册为模型的缓冲区，用于存储因果掩码
        # self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))  # 下三角
        self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))  # 上三角
    
    # def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     将 K/V 的头数从 num_kv_heads 扩展到 num_heads
    #     输入形状: (B, H_kv, T_total, d_k) -> 输出形状: (B, H, T_total, d_k)
    #     """
    #     if self.num_groups == 1:
    #         return x
    #     B, H_kv, T, d_k = x.shape
    #     # 面试加分写法：使用 expand + reshape 比 torch.repeat_interleave(x, dim=1) 更省显存开销
    #     return (
    #         x[:, :, None, :, :]
    #         .expand(B, H_kv, self.num_groups, T, d_k)
    #         .reshape(B, H_kv * self.num_groups, T, d_k)
    #     )

    def forward(self, x, past_key_value= None):
        B, T, D = x.shape  # T: Prefill 阶段为 prompt 长度，Decode 阶段为 1

        # 1. 线性投影 + 拆分多头
        # q: (B, T, D) -> (B, T, H_q, d_k)
        # k, v: (B, T, D) -> (B, T, H_kv, d_k)
        q = self.wq(x).view(B, T, self.num_heads, self.d_k)
        k = self.wk(x).view(B, T, self.num_kv_heads, self.d_k)
        v = self.wv(x).view(B, T, self.num_kv_heads, self.d_k)
        
        # 2. 转置(便于矩阵乘法)
        # q: (B, T, H_q, d_k) -> (B, H_q, T, d_k)
        # k, v: (B, T, H_kv, d_k) -> (B, H_kv, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # # =========应用RoPE旋转位置编码（只作用 Q/K）
        # q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        # k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # =========使用KV缓存
        # GQA 让 KV Cache 缩小到 H_kv/H，提升吞吐量
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # 在序列维度 T 上拼接: (B, H_kv, T_past + T_q, d_k)
            v = torch.cat([past_v, v], dim=2)

        # 3. 对 K 和 V 进行分组扩展 (H_kv -> H_q)
        # [B, H_kv, T, d_k] -> [B, H_q, T, d_k]  (H_q = H_kv * G)
        k = torch.repeat_interleave(k, repeats=self.repetitions, dim=1)
        v = torch.repeat_interleave(v, repeats=self.repetitions, dim=1)

        # 3 缩放点积计算（与 MHA 相同）
        # (B, H_q, T, d_k) @ (B, H_q, d_k, T) -> (B, H_q, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5) 
        
        # 【改动 4】动态构建 Causal Mask（兼顾 Prefill 和 Decode 以及 Chunked Prefill）
        if T > 1:
            scores.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        
        # 4.分数归一化（Softmax）
        attn = F.softmax(scores, dim=-1)
        
        # 5 加权求和
        # (B, H_q, T, T) @ (B, H_q, T, d_k) -> (B, H_q, T, d_k)
        context = attn @ v

        # 合并 heads
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        output =  self.wo(context)
        
        return output, (k, v)  # 返回 KV Cache 以便下次调用


if __name__ == "__main__":
    B, T, D = 2, 64, 512
    num_heads = 8
    num_kv_heads = 2  # 8个Q头，2个KV头，每组4个Q共享1组KV

    x = torch.randn(B, T, D)
    gqa = GroupedQueryAttention(
        d_model=D, 
        num_heads=num_heads, 
        num_kv_heads=num_kv_heads, 
        max_seq_len=512
    )
    out, past_kv = gqa(x)
    
    print("输入维度:", x.shape)
    print("输出维度:", out.shape)
    
    # 验证输出维度
    assert out.shape == (B, T, D), "维度校验失败！"
    print("GQA 实现验证成功！")
