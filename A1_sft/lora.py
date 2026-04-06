import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha  # 权重
        
        # 1. 提取原线性层的维度
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 2. 冻结原模型的权重矩阵 W
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # 3. 初始化低秩矩阵 WA 和 WB
        # WA 使用 Kaiming 初始化，WB 初始化为 0，确保训练开始时旁路输出为 0，不影响原模型性能
        self.WA = nn.Parameter(torch.zeros(self.in_features, rank))
        self.WB = nn.Parameter(torch.zeros(rank, self.out_features))
        
        nn.init.kaiming_uniform_(self.WA, a=math.sqrt(5))
        nn.init.zeros_(self.WB)

    def forward(self, x):
        # 原路径计算：h = XW
        original_output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            original_output += self.bias
            
        # 旁路路径计算：h_lora = X * (WA * WB) * alpha
        # 实际实现中通常先做矩阵乘法 X * WA 再乘 WB 以提高效率
        lora_output = (x @ self.WA @ self.WB) * self.alpha
        
        # 最终输出为两路径之和
        return original_output + lora_output

# 学习测试：如何替换现有模型中的线性层
def apply_lora(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 将原 Linear 层替换为 LoRALinear
            new_layer = LoRALinear(module, rank=rank)
            setattr(model, name, new_layer)
            print(f"已替换层: {name}")
    return model