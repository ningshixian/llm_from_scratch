# torchrun  adam.py

import torch
import torch.nn as nn
import torch.optim as optim

# 模型定义
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.w1 = nn.Linear(128, 512, bias = False)
        self.w2 = nn.Linear(512, 10, bias = False)

    def forward(self, x):
        hidden = self.w1(x)
        output = self.w2(hidden)
        return output, hidden


class MyAdam:
    def __init__(self, params, lr = 1e-3, beta1 = 0.90, beta2 = 0.999,  eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # 初始化一阶和二阶动量缓存
        self.M = [ torch.zeros_like(p.data) for p in self.params ]
        self.V = [ torch.zeros_like(p.data) for p in self.params ]

    @torch.no_grad() # 核心：避免跟踪优化器内部计算图
    def step(self, weight_decay=1e-2):
        self.t += 1
        for i, param in enumerate(self.params):
            grad = param.grad

            # 更新一阶矩 (Momentum) 和 二阶矩 (Variance)
            self.M[i] = self.beta1 * self.M[i] + (1 - self.beta1) * grad
            self.V[i] = self.beta2 * self.V[i] + (1 - self.beta2) * (grad ** 2)
            # 偏差修正
            m_hat = self.M[i] / (1 - self.beta1 ** self.t)
            v_hat = self.V[i] / (1 - self.beta2 ** self.t)
            # 参数更新
            update = m_hat / (torch.sqrt(v_hat) + self.eps)
            if weight_decay > 0:    # AdamW (Weight Decay 实际上是直接作用于参数，不通过梯度)
                param.data.mul_(1 - self.lr * weight_decay)
            # param.data.sub_(self.lr * update)
            param.data -= (self.lr * update)
    
    def zero_grad(self, ):
        for param in self.params:
            param.grad = None


# 训练循环
def train(rank, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        optimizer.zero_grad()      # 1. 清空梯度
        outputs, _ = model(input)  # 2. 前向传播
        loss = loss_fn(outputs, labels) # 3. 计算损失
        loss.backward()            # 4. 反向传播 (DDP 在这里进行梯度同步 AllReduce)
        optimizer.step()           # 5. 更新参数
        
        if rank == 0 and i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss.item():.6f}")


if __name__ == '__main__':
    #设置随机种子以便复现
    torch.manual_seed(42)

    # 准备数据
    N, D_in, D_out = 128, 128, 10
    input = torch.randn(N, D_in)
    labels = torch.randn(N, D_out)

    # 实例化组件
    model = ToyModel()
    loss_fn = nn.MSELoss()
    
    # 使用你手写的 Adam
    optimizer = MyAdam(model.parameters(), lr=0.001)

    print("开始本地训练...")
    train(model, input, labels, loss_fn, optimizer, epochs=1000)
    print("训练结束")
