import torch
from torch import nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups):
        super().__init__()
        # 必须保证能整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.num_heads = num_heads           # Query 头的总数 (例如: 12)
        self.num_kv_groups = num_kv_groups   # KV 头的总数 (例如: 2)
        self.group_size = num_heads // num_kv_groups # 每组包含多少个 Q 头 (例如: 6)
        self.head_dim = d_out // num_heads   # 每个头的维度
        
        # 投影层：注意 KV 的输出维度远小于 Q
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
    
    def forward(self, x):
        b, num_tokens, _ = x.shape

        # 1. 投影与重塑
        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)
        # 变换维度以便计算注意力 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # (b, num_heads, num_tokens, head_dim)

        # 2. 核心：扩展 KV 头以匹配 Q 头 (Key/Value Repeat)
        # 使用 repeat_interleave 确保相邻的 Q 头共享同一个 KV 头
        # 变换后：[b, num_heads, tokens, head_dim]
        keys = keys_new.repeat_interleave(self.group_size, dim=1)
        values = values_new.repeat_interleave(self.group_size, dim=1)

        # 3. 标准点积注意力计算
        attn_scores = queries @ keys.transpose(2, 3) / (self.head_dim**0.5)
        
        # causal mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        # 训练/预填充模式：从 0 开始
        q_positions = torch.arange(num_tokens_Q)
        k_positions = torch.arange(num_tokens_K)
        # 生成因果掩码矩阵(True 表示需要被遮盖的位置)
        # q_positions: [0, 1, 2], k_positions: [0, 1, 2]
        # mask 结果为：
        # [[False, True,  True],  <- 第 0 个 token 只能看自己
        #  [False, False, True],  <- 第 1 个 token 能看 0 和 1
        #  [False, False, False]] <- 第 2 个 token 能全看
        mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)
        # Use the mask to fill attention scores
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # ... 应用 Softmax 和 dropout ...
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = nn.Dropout(0.1)(attn_weights)

        # 4. 输出投影
        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, -1)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec