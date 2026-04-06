"""
RLHF-PPO训练流程伪代码
包含四个关键模型：
1. ref_model: 参考模型，用于KL惩罚
2. reward_model: 奖励模型，评估生成文本质量
3. actor_model: 演员模型，生成文本
4. critic_model: 评论员模型，评估状态价值
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RLHF_PPO:
    def __init__(self):
        # 初始化模型
        self.ref_model = LLM().eval()       # 初始预训练模型
        self.reward_model = LLM().eval()    # 奖励模型（已训练好）
        self.actor_model = LLM()            # 从ref_model复制
        self.critic_model = LLM()           # 价值头网络
        
        # 优化器
        self.actor_opt = Adam(self.actor_model.parameters(), lr=1e-5)
        self.critic_opt = Adam(self.critic_model.parameters(), lr=1e-5)
        
        # 超参数
        self.kl_beta = 0.1       # KL 散度惩罚系数 (beta)
        self.clip_eps = 0.2      # 截断阈值
        # gamma = 0.99                # 折扣因子
        # lam = 0.95                  # GAE 平滑因子
        self.ppo_epochs = 4      # PPO更新轮数
        self.batch_size = 256    # 批次大小
        self.max_steps = 1000    # 最大训练步数
        self.critic_weight = 0.5     # 价值函数损失权重
        # entropy_coef = 0.01         # 熵正则化系数 (鼓励探索)
        
    def train(self, train_data):
        """完整训练流程"""
        for step in range(self.max_steps):
            # 1. 采样阶段：收集轨迹{s,a,r}
            trajectories = self.collect_trajectories(train_data)
            
            # 2. 计算优势和回报{A,G}
            advantages, returns = self.compute_advantages(trajectories)
            
            # 3. PPO更新阶段
            self.ppo_update(trajectories, advantages, returns)
            
    def collect_trajectories(self, train_data):
        """阶段 1: 采样 (Rollout) - 获取经验数据"""
        trajectories = []

        for batch in DataLoader(train_data, batch_size=self.batch_size):
            prompts = batch['input_ids']
            with torch.no_grad():
                # 1. Actor 生成 (Rollout)，包括 Prompt + Response
                seq_ids = self.actor_model.generate(prompts)  # [batch, seq_len]
                # 标记哪些是 Response (1), 哪些是 Prompt/Padding (0)
                action_mask = (seq_ids != 0).long() 

                # 2. 获取 actor/ref 的log概率
                logits = self.actor_model(seq_ids)
                old_log_probs = F.log_softmax(logits, dim=-1)
                ref_logits = self.ref_model(seq_ids)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                # 3. 获取价值估计 (Value)
                old_values = self.critic_model(seq_ids).squeeze(-1)
            
                # 4. 计算奖励 (Reward + KL Penalty)
                # 原始奖励 (Scalar) → 只在最后一个 token 有值，前面都是 0
                raw_rewards = self.reward_model(seq_ids)    #[batch, 1] 的标量
                # KL 惩罚 (Per Token) → 将 KL 惩罚应用到每一步的奖励中
                kl_penalty = self.kl_beta * (old_log_probs - ref_log_probs)
                rewards = -kl_penalty
                last_idxs = action_mask.sum(dim=1) - 1
                rewards[:, last_idxs] += raw_rewards.view(-1)

                # 修正：Mask 掉 Prompt 部分的奖励和 KL，防止干扰
                rewards = rewards * action_mask 
            
            # 存储轨迹
            trajectories.append({
                'input_ids': prompts,
                'seq_ids': seq_ids,
                'old_log_probs': old_log_probs, # [B, L]
                'rewards': rewards,             # [B, L]
                'values': old_values,           # [B, L]
                'masks': action_mask            # [B, L]
            })
            
        return trajectories
    
    def compute_advantages(self, trajectories):
        """计算优势 (Advantage) 和 回报 (Return)"""
        advantages = []
        returns = []

        for traj in trajectories:
            # 注意：GAE 计算时也要考虑 action_mask，不要把 padding 算进去
            adv = self.compute_gae(traj['rewards'], traj['old_values'], ...)
            # Return = Advantage + Value
            ret = adv + traj['old_values']
            
            advantages.append(adv)
            returns.append(ret)
            
        return advantages, returns
    
    def ppo_update(self, trajectories, advantages, returns):
        """PPO更新阶段"""
        # PPO 通常会在同一批数据上更新多次 (Epochs)
        for _ in range(self.ppo_epochs):
            for traj, advantage, return_ in zip(trajectories, advantages, returns):
                # 1. 重新计算 Log Probs 和 Values (从Epoch 2 始，θ参数已更新)
                logits = self.actor_model(traj['seq_ids'])
                new_log_probs = F.log_softmax(logits, dim=-1)
                new_values = self.critic_model(traj['seq_ids']).squeeze(-1)

                # 2. Masking (关键：只计算 Response 部分的 Loss)
                mask = traj['action_mask']
                
                # --- A. Policy Loss (Actor Loss) ---
                # 计算概率比率 r(t) = exp(new_log - old_log)
                ratio = torch.exp(new_log_probs - traj['old_log_probs'])
                
                # PPO 核心公式: min(ratio * A, clip(ratio) * A)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
                actor_loss = -torch.min(surr1, surr2)
                actor_loss = (actor_loss * mask).sum() / mask.sum()
                
                # --- B. Value Loss (Critic Loss) ---
                critic_loss = F.mse_loss(new_values, return_)
                critic_loss = (critic_loss * mask).sum() / mask.sum()

                # --- C. 总 Loss 联合训练 ---
                loss = actor_loss + self.critic_weight * critic_loss

                # 4. 反向传播和优化
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()         # ？
                self.actor_opt.step()
                self.critic_opt.step()
