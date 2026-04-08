import torch
import torch.nn.functional as F

# 温度采样 - 通过温度参数调整概率分布
def sample(logits, temperature: float = 1.0):
    assert temperature > 0, "Temperature must be greater than 0"
    
    logits = logits / max(temperature, 1e-5)
    probs = F.softmax(logits, dim=-1)
    
    # 从分布中采样
    # next_token = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_token

