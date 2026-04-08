import numpy as np

def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply LayerNorm to the input array.

    Parameters:
        x   : np.ndarray of shape (batch_size, features)
        eps : float - small constant for numerical stability

    Returns:
        np.ndarray of same shape as x
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply RMSNorm to the input array.
    
    Parameters:
        x   : np.ndarray of shape (batch_size, features)
        g   : np.ndarray of shape (features,) - gain parameter
        eps : float - small constant for numerical stability
    
    Returns:
        np.ndarray of same shape as x
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * g