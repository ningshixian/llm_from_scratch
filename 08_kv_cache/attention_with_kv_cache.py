import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)


# 面试白板手撕
def single_attention_step_with_kv_cache(
    x_t: torch.Tensor,          # shape: (1, d_model) 当前时刻输入
    W_q: torch.Tensor,          # shape: (d_model, d_head)
    W_k: torch.Tensor,          # shape: (d_model, d_head)
    W_v: torch.Tensor,          # shape: (d_model, d_head)
    k_cache: torch.Tensor,      # shape: (t-1, d_head)
    v_cache: torch.Tensor       # shape: (t-1, d_head)
):
    """
    单步/单步自注意力 + KV cache
    返回:
        output: (1, d_head)
        new_k_cache: (t, d_head)
        new_v_cache: (t, d_head)
    """
    # 1) 计算当前 token 的 q, k, v
    q_t = x_t @ W_q   # (1, d_head)
    k_t = x_t @ W_k   # (1, d_head)
    v_t = x_t @ W_v   # (1, d_head)

    # 2) 把当前 token 的 k, v 拼到 cache 里
    if k_cache is None:
        new_k_cache = k_t
        new_v_cache = v_t
    else:
        new_k_cache = torch.cat([k_cache, k_t], dim=0)  # (t, d_head)
        new_v_cache = torch.cat([v_cache, v_t], dim=0)  # (t, d_head)

    # 3) 用当前 q_t 和全部历史 K 做打分
    #    (1, d_head) @ (d_head, t) -> (1, t)
    scores = q_t @ new_k_cache.transpose(0, 1) / math.sqrt(q_t.size(-1))

    # 4) softmax 得到注意力权重
    attn_weights = torch.softmax(scores, dim=-1)  # (1, t)

    # 5) 对全部历史 V 加权求和
    #    (1, t) @ (t, d_head) -> (1, d_head)
    output = attn_weights @ new_v_cache

    return output, new_k_cache, new_v_cache


# 工程优化版
class AttentionKVCache(nn.Module):
    def __init__(self, dim = 512):
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.k_cache = None   # shape: (1, t, dim)
        self.v_cache = None   # shape: (1, t, dim)
        
    def forward(self, x, mask, verbose = False):
        # 单步 attention，输入只有当前 1 个 token。
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # 更新 cache ----------
        if self.k_cache is None:
            self.k_cache = k   # (1, 1, dim)
            self.v_cache = v   # (1, 1, dim)
        else:
            # 按 seq_len 维拼接
            self.k_cache = torch.cat([self.k_cache, k], dim=1)   # (1, t, dim)
            self.v_cache = torch.cat([self.v_cache, v], dim=1)   # (1, t, dim)
        k,v = self.k_cache, self.v_cache
        
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim)

        # 单步 KV cache 推理时，不需要 causal mask：
        # 因为 cache 里只有历史和当前，不含未来 token

        attn_weights = F.softmax(scores, dim = -1)
        out = attn_weights @ v
        return self.wo(out)
