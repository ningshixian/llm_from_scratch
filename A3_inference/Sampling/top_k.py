import torch

def sample(logits: torch.Tensor, top_k: int = None):
    """
    Sample from logits
    """
    # New logits filled with $-\infty$; i.e. zero probability
    zeros = logits.new_ones(logits.shape) * float('-inf')
    # Apply top-k sampling
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, top_k_indices, top_k_logits)
        
    # Sample from the top-k logits with the specified sampler.
    return logits