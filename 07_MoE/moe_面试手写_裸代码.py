import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([FFN(dim) for _ in range(num_experts)])  # 复用下面的 FFN

    def forward(self, x):
        B, T, d = x.shape
        x = x.view(-1, d)                                     # [N, d] 摊平

        # --- router: 返回 topk 权重 + idx ---
        logits = self.gate(x)                                 # [N, E]
        w, idx = logits.softmax(-1).topk(self.top_k, dim=-1)  # [N, k] 权重 + 选中的专家
        w = w / w.sum(-1, keepdim=True)                       # top-k 内重归一化

        # --- dispatch: per expert ---
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                  # 循环专家 (E 次), 不循环 token
            token_idx, token_pos = (idx == e).nonzero(as_tuple=True) # 哪个 token / 它的第几个 top-k 选择
            if len(token_idx) == 0:
                continue
            # x[token_idx]        "=" 右边：选出来的密集子集拷贝
            # out[token_idx]  "=" 左边：稀疏地修改若干行
            out[token_idx] += w[token_idx, token_pos].view(-1, 1) * expert(x[token_idx])
        return out.view(B, T, d), logits                           # logits 带出去算 aux loss


# 追问再补 FFN（Llama SwiGLU 版，可直接 call 不展开）：
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
