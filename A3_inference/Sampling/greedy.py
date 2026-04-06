import torch

def sample(logits: torch.Tensor):
    """
    Sample the most likely token from the distribution of logits
    """
    return logits.argmax(dim=-1)
