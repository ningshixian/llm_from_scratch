import torch
import torch.nn as nn
import torch.nn.functional as F

"""因果注意力 / 掩码注意力"""

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, max_seq_len, qkv_bias=False):
        super().__init__()
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        # # 注册为模型的缓冲区，用于存储因果掩码
        # self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))  # 下三角
        self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))  # 上三角

    def forward(self, x):
        # x 维度: (batch_size, seq_len, embed_dim)
        B, T, D = x.shape

        # 1. 映射得到 Q, K, V
        # 等价于 x @ W_q + b_q
        Q = self.W_q(x)  # (B, L, D)
        K = self.W_k(x)    # (B, L, D)
        V = self.W_v(x)  # (B, L, D)

        # 2. 计算注意力得分 (Scores)
        # (B, L, D) @ (B, D, L) -> (B, L, L)
        scores = Q @ K.transpose(-2, -1) / (D ** 0.5)

        # ========== NEW：因果掩码 ============
        # scores.masked_fill_(self.mask == 0, -torch.inf)
        scores.masked_fill_(self.mask.bool()[:T, :T], -torch.inf)

        # 4. 归一化 (Softmax)
        # 在每一行进行 Softmax，使每一行权重和为 1
        attn_weights = F.softmax(scores, dim=-1)

        # 5. 加权求和 (Output)
        # (B, L, L) @ (B, L, D) -> (B, L, D)
        context_vec = attn_weights @ V
        
        return context_vec, attn_weights


# --- 测试代码 ---
torch.manual_seed(123)
d_in, d_out = 4, 4
seq_length = 3  # 假设输入句子有 3 个单词
input_data = torch.randn(1, seq_length, d_in)

model = SelfAttention(d_in, d_out, max_seq_len=seq_length)
output, weights = model(input_data)

print("输入维度:", input_data.shape)
print("输出维度:", output.shape)
print("注意力权重矩阵:\n", weights)