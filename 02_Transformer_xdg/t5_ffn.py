import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim    # 通常 FFN 的隐藏层维度是输入维度的 4 倍
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        # 公式: FFN(x) = W2(ReLU(W1(x)))
        return self.w2(F.relu(self.w1(x)))
