# https://github.com/dhcode-cpp/grpo-loss

import torch
import torch.nn.functional as F

def compute_group_advantages(rewards, epsilon: float = 1e-8):
    """
    计算组内相对优势 (Group Relative Advantages)。
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean) / (std + epsilon)
    return advantages

def kl_divergence_estimator(log_probs, ref_log_probs):
    kl_log_ratio = ref_log_probs - log_probs 
    kl_penalty = torch.exp(kl_log_ratio) - kl_log_ratio - 1
    return kl_penalty

def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,		    # 序列级别的奖励 (Batch*G,)
    attention_mask: torch.Tensor,   # 掩码 (Batch*G, Seq_Len)，1为有效token，0为padding
    group_size: int,
    beta: float = 0.01,			# KL 散度的系数
    clip_eps: float = 0.2		# PPO 裁剪系数
) -> torch.Tensor:

    rewards = rewards.view(-1, group_size)	# 重塑为 (Batch_Size, Group_Size) 以便进行组内计算
    advantages = compute_group_advantages(rewards)
    advantages = advantages.unsqueeze(dim = 1) # [a, b ,c] -> [[a], [b], [c]]

    # 重要性采样比率 (Ratio)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    # PPO Clipped Surrogate Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = torch.minimum(surr1, surr2) # 形状: (Batch*G, Seq_Len)

    # KL 近似估计
    kl_penalty = kl_divergence_estimator(log_probs, ref_log_probs)

    # Loss = -PolicyLoss + beta * KL
    token_loss = -policy_loss + beta * kl_penalty
    masked_loss = token_loss * attention_mask			# 屏蔽 Padding Token
    loss = masked_loss.sum() / attention_mask.sum()		# 只对有效 Token 求平均
    return loss