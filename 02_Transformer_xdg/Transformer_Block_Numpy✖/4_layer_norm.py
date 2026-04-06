import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    """
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    norm_X = gamma * X_norm + beta
    return norm_X


def rms_normalization(x, gamma, epsilon=1e-6):
    # 1. 计算均方根 (RMS)
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + epsilon)
    # 2. 归一化并缩放
    return (x / rms) * gamma
