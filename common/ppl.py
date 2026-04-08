import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    PPL=exp(−N1​i=1∑N​logpi​)
    """
    log_sum = 0.0
    n = len(probabilities)
    
    for p in probabilities:
        if p <= 0 or p > 1:
            raise ValueError("each probability must be in (0, 1]")
        log_sum += np.log(p)
    
    return np.exp(-log_sum / n)
