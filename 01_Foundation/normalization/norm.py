# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Transformer 层数一深，中间激活的尺度就会不断变化。如果不做归一化，后面的矩阵乘法和激活函数很容易面对不稳定的输入；但标准 LayerNorm 又要同时计算均值和方差，在大模型里会带来额外开销。

RMSNorm 选择更直接的做法：不再减均值，只用均方根控制特征尺度。本节会实现一个最小 RMSNorm，重点看清归一化维度、eps 的数值稳定作用，以及为什么许多 LLaMA 系模型会采用这种更轻的归一化层。完成后，你可以把它接到后面的 MLP、注意力和完整 Transformer 组件里。
"""


def batch_norm(x, gamma, beta, eps=1e-5):
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    normalized = (x - batch_mean) / torch.sqrt(batch_var + eps)
    return normalized * gamma + beta, batch_mean, batch_var


def layer_norm_last_dim(x, gamma, beta, eps=1e-5):
    """减均值、按标准差缩放"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized * gamma + beta


def rms_norm_last_dim(x, weight, eps=1e-6):
    """不执行均值中心化、按均方根缩放"""
    # torch.rsqrt 是平方根倒数，效率更高
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(mean_square + eps)
    return normalized * weight


x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
gamma = torch.ones(3)
beta = torch.zeros(3)
print('BatchNorm 输出：')
y, mean, var = batch_norm(x, gamma, beta)
print(y)
print('LayerNorm 输出：')
print(layer_norm_last_dim(x, gamma, beta))


x = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]])
weight = torch.ones(3)
layer_output = layer_norm_last_dim(x, weight, torch.zeros(3), eps=1e-6)
rms_output = rms_norm_last_dim(x, weight, eps=1e-6)

print('LayerNorm row mean:', layer_output.mean(dim=-1).tolist())
print('RMSNorm row mean:', rms_output.mean(dim=-1).tolist())
print('RMSNorm row RMS:', rms_output.pow(2).mean(dim=-1).sqrt().tolist())