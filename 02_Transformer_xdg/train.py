import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 从 t6_transformer.py 引入模型和配置参数类
from t6_transformer import Transformer, ModelArgs

# 简单的中文训练语料
CORPUS = [
    "我喜欢学习人工智能",
    "深度学习改变世界",
    "今天天气非常晴朗",
    "自然语言处理很有趣",
    "模型正在逐步收敛",
    "祝你今天工作顺利",
    "快乐地探索大模型技术",
    "编程是一件有趣的事情",
    "代码运行完全正常",
    "梯度下降优化网络参数"
]

# 1. 构造一个轻量的模拟序列数据集
class ToySeqDataset(Dataset):
    """
    构造简单的序列任务（例如输入序列，目标为向后位移一位的自回归标签）
    """
    def __init__(self, texts, tokenizer, max_len=16):
        self.data = []

        for text in texts:
            # 使用 BertTokenizer 分词并填充/截断到固定长度
            encoding = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            src_idx = encoding['input_ids'].squeeze(0)
            
            # target_idx: 自回归预测下一个字符（右移一位）
            # ignore_index 设置为 -100，同时将原有的 PAD 位置也标记为 -100
            target_idx = torch.full_like(src_idx, -100)
            target_idx[:-1] = src_idx[1:]
            
            # 将 PAD 对应的预测位置也忽略
            pad_mask = (target_idx == tokenizer.pad_token_id)
            target_idx[pad_mask] = -100
            
            self.data.append((src_idx, target_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train():
    # 2. 设置设备与随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 3. 实例化 t6_transformer.py 所需的超参数
    args = ModelArgs(
        n_embd=64,       # 词嵌入维度 (需满足 dim % n_heads == 0)
        n_heads=4,       # 多头注意力头数
        dim=64,          # 隐藏层维度
        dropout=0.1,     # Dropout 概率
        max_len=128,     # 支持的最大序列长度
        vocab_size=500,  # 词表大小
        n_layer=2        # Encoder 和 Decoder 的层数
    )
    
    # 加载 BertTokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    args.vocab_size = tokenizer.vocab_size

    # 4. 初始化模型
    model = Transformer(args).to(device)

    # 5. 准备数据加载器
    dataset = ToySeqDataset(texts=CORPUS, tokenizer=tokenizer, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # 6. 配置优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # 7. 训练循环
    epochs = 5
    print(f"开始训练，设备: {device}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step, (src_idx, target_idx) in enumerate(dataloader):
            src_idx = src_idx.to(device)
            target_idx = target_idx.to(device)

            optimizer.zero_grad()

            # t6_transformer 的 forward 传入 target_idx 时内部直接计算好交叉熵损失
            logits, loss = model(src_idx=src_idx, target_idx=target_idx)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | Step [{step + 1}/{len(dataloader)}] | Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch [{epoch + 1}/{epochs}] 完成，平均 Loss: {avg_loss:.4f} ---\n")

    # 8. 保存模型权重
    torch.save(model.state_dict(), "t6_transformer_checkpoint.pth")
    print("模型权重保存完成: t6_transformer_checkpoint.pth")


if __name__ == "__main__":
    train()