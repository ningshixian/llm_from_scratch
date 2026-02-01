import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 定义三个线性变换矩阵：Query, Key, Value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x 维度: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()

        # 1. 映射得到 Q, K, V
        Q = self.query(x)  # (B, L, D)
        K = self.key(x)    # (B, L, D)
        V = self.value(x)  # (B, L, D)

        # 2. 计算注意力得分 (Scores)
        # Q 与 K 的转置相乘，衡量每个词之间的相关性
        # (B, L, D) @ (B, D, L) -> (B, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) 

        # 3. 缩放 (Scaling)
        # 防止维度过大导致 Softmax 梯度消失
        scores = scores / (embed_dim ** 0.5)

        # NEW：因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)

        # 4. 归一化 (Softmax)
        # 在每一行进行 Softmax，使每一行权重和为 1
        attention_weights = F.softmax(scores, dim=-1)

        # 5. 加权求和 (Output)
        # 使用权重对 V 进行加权
        # (B, L, L) @ (B, L, D) -> (B, L, D)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

# --- 测试代码 ---
embed_size = 4
seq_length = 3  # 假设输入句子有 3 个单词
input_data = torch.randn(1, seq_length, embed_size)

model = SimplifiedSelfAttention(embed_size)
output, weights = model(input_data)

print("输入维度:", input_data.shape)
print("输出维度:", output.shape)
print("注意力权重矩阵:\n", weights)