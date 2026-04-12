import torch

def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        top_k_val = logits.topk(top_k).values[-1]
        logits[logits < top_k_val] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = (cumsum - probs) > top_p
        sorted_logits[mask] = float('-inf')
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


# Demo
logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
print('top_k=1:', sample_top_k_top_p(logits.clone(), top_k=1))
print('top_p=0.5:', sample_top_k_top_p(logits.clone(), top_p=0.5))