import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even."

        # Empty encodings vectors
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2)
        div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))

        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # 注册成 buffer，默认不可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        # x: (batch_size, seq_len, d_model)
        pe = self.pe[:, :x.size(1)]
        return x + pe


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    pos_enc = PositionalEncoding(d_model=20, max_len=100)

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(100), pos_enc.pe[0, :, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title("Positional encoding")
    plt.show()