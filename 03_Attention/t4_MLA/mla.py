import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
MHA 保存每个 KV Head 的历史状态，GQA/MQA 通过共享 KV Head 减少重复；MLA 进一步改变历史状态的表示，把可压缩内容写入低维 latent，并单独保留位置相关分量。
"""


class MultiHeadsLatentAttention(nn.Module):
    def __init__(self, dim, n_heads, d_latenT, context_length):
        super().__init__()
        
        self.d_model = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # 简化的 head_dim
        self.d_latent = d_latent        # 压缩后的低秩维度 (c_kv)

        # --- 1. KV 压缩部分 ---
        # 降维投影：将输入投影到低秩潜空间
        self.W_DKV = nn.Linear(dim, d_latenT, bias=False)
        
        # 升维投影：从潜空间还原 Q, K, V
        self.W_UK = nn.Linear(d_latenT, dim, bias=False) 
        self.W_UV = nn.Linear(d_latenT, dim, bias=False)
        self.W_UQ = nn.Linear(dim, dim, bias=False)

        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(0.1)

        # # 注册为模型的缓冲区，用于存储因果掩码
        # self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))  # 下三角
        # self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))  # 上三角

    def forward(self, x):
        B, T, D = x.shape  # T: Prefill 阶段为 prompt 长度，Decode 阶段为 1
        
        # [步骤 A] KV 压缩投影
        # 这里的 c_kv 是唯一需要存入 KV Cache 的向量，极大地减少了存储量
        c_kv = self.W_DKV(x) # (B, T, d_latent)

        # [步骤 B] 还原 K, V 向量并分头
        q = self.W_UQ(x).view(B, T, self.n_heads, self.head_dim)
        k = self.W_UK(c_kv).view(B, T, self.n_heads, self.head_dim)
        v = self.W_UV(c_kv).view(B, T, self.n_heads, self.head_dim)

        # [步骤 C] KV Cache 拼接
        # 这里的 KV Cache 只需要存储 c_kv，而不是完整的 K, V 向量，节省了大量内存
        # 例如：原始 MHA 的 KV Cache 维度为 (B, H, T_total, head_dim
        # MLA 的 KV Cache 维度为 (B, T_total, d_latent)

        # [步骤 D] 缩放点积注意力
        # 注意：这里的 head_dim 变成了原维度 + RoPE 维度
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)
        
        # 4. 应用因果掩码 (将未来位置的分数设为负无穷)
        scores.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 5 加权求和
        # (B, H_q, T, T) @ (B, H_q, T, d_k) -> (B, H_q, T, d_k)
        context = attn @ v
        
        # [步骤 E] 计算上下文向量
        # 注意：Value 向量是不带 RoPE 信息的
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        context_vec = self.wo(context)  # optional projection
        
        return context_vec


# --- 学习用测试代码 ---
if __name__ == "__main__":
    max_len = 1024
    cfg = {"dim": 64, "n_heads": 8, "d_latent": 16, "context_length": max_len}
    model = MultiHeadsLatentAttention(**cfg)
    sample_input = torch.randn(1, 10, 64)
    output = model(sample_input)
    print(f"Input Shape: {sample_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"MLA 成功运行！KV Cache 存储维度从 {2 * cfg['dim']} 降至 {cfg['d_latent'] + cfg['d_rope']}")