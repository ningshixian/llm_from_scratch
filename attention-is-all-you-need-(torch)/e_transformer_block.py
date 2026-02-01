import torch
import torch.nn as nn

from c_multi_head_self_attention import MultiHeadAttention
from d_ffn import MLP, RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim=None):
        super().__init__()
        
        # 1. 注意力层及其预归一化
        self.attention_norm = RMSNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # 2. 前馈网络层及其预归一化
        # 注：你之前的 ffn.py 中 FFN 类已经内置了 RMSNorm 和 残差连接
        # 为了展示清晰，这里我们手动展示 Block 的内部逻辑
        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = MLP(embed_dim, hidden_dim) # 使用你定义的 MLP 类

    def forward(self, x):
        """
        x 维度: (batch_size, seq_len, embed_dim)
        """
        h = x + self.attention(self.attention_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out

# --- 快速测试 ---
if __name__ == "__main__":
    # 参数设置
    dim = 128
    heads = 8
    seq_len = 10
    batch = 2
    
    # 初始化模型
    block = TransformerBlock(embed_dim=dim, num_heads=heads)
    
    # 模拟输入
    x = torch.randn(batch, seq_len, dim)
    
    # 前向传播
    output = block(x)
    
    print(f"Transformer Block 输入形状: {x.shape}")
    print(f"Transformer Block 输出形状: {output.shape}") 
    # 输出应该是 [2, 10, 128]，与输入保持一致，方便堆叠