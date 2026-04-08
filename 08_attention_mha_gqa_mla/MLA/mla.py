import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadsLatentAttention(nn.Module):
    def __init__(self, dim, n_heads, d_latent, context_length):
        super().__init__()
        self.n_heads = n_heads
        self.d_out = dim
        self.head_dim = dim // n_heads  # 简化的 head_dim
        self.d_latent = d_latent        # 压缩后的低秩维度 (c_kv)

        # --- 1. KV 压缩部分 ---
        # 降维投影：将输入投影到低秩潜空间
        self.W_DKV = nn.Linear(dim, d_latent, bias=False)
        
        # 升维投影：从潜空间还原 Q, K, V
        # 学习建议：推理阶段 W_UK 可以被吸收进 Q 投影，W_UV 可以被吸收进 Out_proj
        self.W_UK = nn.Linear(d_latent, dim, bias=False) 
        self.W_UV = nn.Linear(d_latent, dim, bias=False)
        self.W_UQ = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(0.1)

        # 注册因果掩码 (防止当前位置看到未来的 token)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, t, d = x.shape
        
        # [步骤 A] KV 压缩投影
        # 这里的 c_kv 是唯一需要存入 KV Cache 的向量，极大地减少了存储量
        c_kv = self.W_DKV(x) # (b, t, d_latent)

        # [步骤 B] 还原 K, V 向量并分头
        keys = self.W_UK(c_kv).view(b, t, self.n_heads, self.head_dim)
        values = self.W_UV(c_kv).view(b, t, self.n_heads, self.head_dim)
        queries = self.W_UQ(x).view(b, t, self.n_heads, self.head_dim)

        # [步骤 D] 缩放点积注意力
        # 注意：这里的 head_dim 变成了原维度 + RoPE 维度
        attn_weights = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. 应用因果掩码 (将未来位置的分数设为负无穷)
        mask_bool = self.mask.bool()[:t, :t]
        attn_weights = attn_weights.masked_fill_(mask_bool, -torch.inf)
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # [步骤 E] 计算上下文向量
        # 注意：Value 向量是不带 RoPE 信息的
        context = (attn_probs @ values.transpose(1, 2)).transpose(1, 2).contiguous()
        context = context.view(b, t, self.d_out)

        context_vec = self.out_proj(context)  # optional projection
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