import torch

# pytorch实现
def cross_entropy_loss(logits, targets):
    log_sum_exp = torch.logsumexp(logits, dim=1)  # (B,)
    true_logits = logits[torch.arange(logits.size(0)), targets]  # (B,)
    loss = -true_logits + log_sum_exp
    return loss.mean()


# import numpy as np

# # numpy实现
# def cross_entropy_loss(logits, targets):
#     # 1. 直接算 log-sum-exp（不做数值稳定处理）
#     log_sum_exp = np.log(np.sum(np.exp(logits), axis=1))  # (B,)
    
#     # 2. 取正确类别的 logit
#     true_logits = logits[np.arange(logits.shape[0]), targets]  # (B,)
#     # logits[:, targets]  # 错！会变成选列，而不是逐行选
    
#     # 3. loss
#     # 标准交叉熵公式是： -log( exp(正确得分) / sum(exp(所有得分)) )
#     # 根据对数除法律：log(A / B) = log(A) - log(B)
#     # 上面的公式瞬间化简为： -( 正确得分 - log_sum_exp )
#     # 再把负号乘进去，就变成了： log_sum_exp - 正确得分
#     loss = -true_logits + log_sum_exp
    
#     return loss.mean()

logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets))
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets))