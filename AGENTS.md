# 项目定位

这是一个用于学习 LLM 算法原理、进行工程实验和准备技术面试的个人仓库。项目重点不是构建单一生产应用，而是通过可阅读、可运行、可对照的代码，从零理解大语言模型的核心组件、主流架构、训练流程、对齐算法和评测方法。

后续在本仓库中工作时，应优先保证：

1. 原理正确，代码能与数学公式对应。
2. 实现清晰，适合逐行学习和面试手撕。
3. 张量形状、掩码方向、归一化维度等关键细节有明确说明。
4. 修改后使用小规模、低成本实验验证，不默认启动大模型下载或长时间训练。

# 仓库性质

- 仓库由多个相对独立的教学模块组成，没有统一的根目录 Python 包、统一依赖文件或统一测试入口。
- 主要语言是 Python，核心框架是 PyTorch；部分基础算法使用 NumPy 实现。
- 大量知识以 Jupyter Notebook、Markdown、PDF 和独立脚本的形式保存。
- 代码既包含个人的“手撕/从零实现”，也包含用于研究前沿架构的完整参考项目或上游代码副本。
- 依赖随模块而变化，常见依赖包括 `torch`、`numpy`、`transformers`、`datasets`、`tiktoken`、`tokenizers`、`trl`、`peft`、`deepspeed`、`accelerate`、`pytest` 等。

# 编码与讲解约定

## 教学优先

- 保留从零实现的核心计算步骤，不要为了代码短而直接替换成高级封装。
- 若使用 `torch.nn.functional.scaled_dot_product_attention`、Transformers Trainer 等现成接口，应说明它对应手写实现中的哪些步骤。
- 新增复杂模块时，优先采用“公式/目的 → 输入输出形状 → 核心步骤 → 最小示例”的结构。
- 注释使用中文为主，关键术语可保留英文，例如 causal mask、logits、KV Cache、advantage。

## 张量形状

- Attention 代码应明确常用符号：`B` 为 batch size，`T/S` 为序列长度，`D` 为隐藏维度，`H` 为头数，`Dh` 为 head dimension。
- 对 `view/reshape`、`transpose/permute`、广播、KV head 复制或压缩等操作，说明变换前后的形状。
- 新增断言以验证 `hidden_size % num_heads == 0`、Q/K/V 维度兼容、mask 可广播等关键前提。

## 数值正确性

- Softmax 应沿 key/token 维度进行，通常是最后一维。
- Causal mask 必须阻止当前位置访问未来 token；注意布尔 mask 与加性 `-inf` mask 的语义差异。
- 交叉熵输入通常整理为 `[B*T, vocab_size]`，标签为 `[B*T]`；padding 标签应使用明确的 `ignore_index`。
- LayerNorm、RMSNorm、RoPE、KL、优势标准化等实现应明确 `eps`、方差定义、缩放因子和 dtype。
- 比较缓存与非缓存生成时，应先验证 logits 或生成 token 一致，再比较速度和显存。

## 可复现性与设备

- 小实验设置随机种子；涉及 CUDA 时同时设置 CUDA seed。
- 设备选择优先遵循 `CUDA → MPS → CPU` 的回退方式，但不要假定所有算子都支持 MPS。
- 测试和示例优先使用 tiny config、小词表、短序列和少量层，避免无意占满显存或内存。
- 不要默认下载大型模型、数据集或权重；需要网络、凭证、GPU 或大量磁盘空间时，先明确告知。


# 面试准备要求

当用户要求讲解或整理面试题时，回答应尽量包含：

1. 一句话定义。
2. 核心公式或伪代码。
3. 张量形状或数据流。
4. 为什么这样设计，以及解决了什么问题。
5. 时间、空间或 KV Cache 复杂度。
6. 与相近方法的对比，例如 MHA/MQA/GQA/MLA、LayerNorm/RMSNorm、PPO/DPO/GRPO。
7. 常见实现陷阱。
8. 可能的追问及简洁答案。
9. 本仓库中对应的代码位置。
10. 向用户解释改了什么、为什么正确、如何自行复现，以及相关面试考点。
