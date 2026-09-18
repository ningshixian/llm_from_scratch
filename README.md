## LLM 算法原理学习和工程实验仓库

[llm_from_scratch](https://github.com/ningshixian/llm_from_scratch)：LLM 算法原理和工程实验仓库，包括 Transformer 的基本组件实现、KV 缓存、**GPT 类主流大模型架构、minimind-LLM 全阶段极简复现、Post-Traning（SFT、DPO、GRPO...）、**

* [bpe](https://github.com/ningshixian/llm_from_scratch/tree/main/bpe)：学习和复现了BPE tokenizer
* [position](https://github.com/ningshixian/llm_from_scratch/tree/main/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81)：常见的位置编码实现 RoPE、YaRN
* [attention](https://github.com/ningshixian/llm_from_scratch/tree/main/attention)：一些常见的注意力机制 MHA/GQA/MLA
* [kv-cache](https://github.com/ningshixian/llm_from_scratch/tree/main/kv-cache)：学习 KV 缓存是如何用空间换时间滴.
* [moe](https://github.com/ningshixian/llm_from_scratch/tree/main/moe)：了解下 MOE 架构的实现
* [gpt](https://github.com/ningshixian/llm_from_scratch/tree/main/gpt)、[llama](https://github.com/ningshixian/llm_from_scratch/tree/main/llama)、[olmo3](https://github.com/ningshixian/llm_from_scratch/tree/main/olmo3)、[qwen3](https://github.com/ningshixian/llm_from_scratch/tree/main/qwen3)：**复现一些 decoder-only的 GPT 类热门模型**
* [SFT](https://github.com/ningshixian/llm_from_scratch/tree/main/sft)：包括**指令微调、LoRA微调.....**
* [DPO](https://github.com/ningshixian/llm_from_scratch/tree/main/dpo)、[GRPO](https://github.com/ningshixian/llm_from_scratch/tree/main/grpo)、[KTO](https://github.com/ningshixian/llm_from_scratch/tree/main/kto)、[PPO](https://github.com/ningshixian/llm_from_scratch/tree/main/ppo)、[reinforce++](https://github.com/ningshixian/llm_from_scratch/tree/main/reinforce%2B%2B)
* [QwenGRPO.ipynb](https://github.com/ningshixian/llm_from_scratch/blob/main/grpo/QwenGRPO.ipynb)：基于通义千问2.5的0.5B模型，复现DeepSeek R1的顿悟时刻
* [minimind-LLM全阶段极简复现](<https://github.com/ningshixian/llm_from_scratch/tree/main/%23minimind-LLM%E5%85%A8%E9%98%B6%E6%AE%B5%E6%9E%81%E7%AE%80%E5%A4%8D%E7%8E%B0%20>) **：** **很 nice 的一个项目，实现了一个极简结构的 LLM，以及全阶段的训练过程：数据集清洗、预训练(Pretrain)、监督微调(SFT)、LoRA微调、直接偏好优化(DPO)、强化学习训练(RLAIF: PPO/GRPO等)

# 目录结构与学习路线

## 00：PyTorch 参考资料

`00_pytorch手册/` 保存 PyTorch 速查表和常用函数手册，主要作为学习时的辅助资料。

## 01：基础组件

`01_Foundation/` 用于理解构成 LLM 的基础算法：

- `activations/`：激活函数及相关 Notebook。
- `losses/`：交叉熵、梯度推导和困惑度相关基础。
- `normalization/`：LayerNorm、RMSNorm 的手动实现和对比分析。
- `optimizers/`：AdaGrad、Adam、Muon 等优化器的实现与总结。
- `position/`：正弦位置编码、RoPE、YaRN 及长度外推实验。
- `tokenization/`：BPE 从零实现，以及 Hugging Face Tokenizers、tiktoken 的使用实验。

## 02：Transformer

`02_Transformer_xdg/` 同时覆盖组件级实现和完整 Encoder-Decoder Transformer：

- `Transformer_Block_Numpy/`：使用 NumPy 实现 Self-Attention、MHA、因果掩码、LayerNorm、RMSNorm 和位置编码，适合推导和面试手撕。
- `Transformer_Block_PyTorch✔/`：使用 PyTorch 分步骤实现 Self-Attention、Causal Attention、MHA、Norm、FFN 和 Transformer Block。
- `transformer.py`：完整 Encoder、Decoder、Cross-Attention、位置编码和 Transformer 模型。
- `main.py`：字符级 tokenizer、翻译数据集、训练、验证、checkpoint 和生成流程；默认数据位于 `notebooks/data.json`，默认输出位于 `output/`。
- `notebooks/`：按框架、Attention、位置编码、LayerNorm、完整 Transformer 的顺序组织学习内容。

## 03—07：经典与主流模型架构

- `03_BERT/`：BERT 原理笔记与 Transformers 使用示例。
- `04_GPT/`：GPT 风格 Decoder-only 模型，包括因果 MHA、LayerNorm、GELU、FFN、TransformerBlock、文本生成及 KV Cache 版本。
- `05_ Llama2_llm/`：Llama 风格实现，核心组件包括 RMSNorm、RoPE、Grouped Query Attention、SwiGLU/FFN 和 Decoder Layer。注意该目录名包含空格，shell 命令中必须正确引用。
- `06_deepseek_v3_xdg/`：DeepSeek-V3 架构研究，覆盖 MLA、MoE、路由与负载均衡、YaRN、多 Token 预测、并行线性层及 FP8 相关接口。`model.py` 引用了 `kernel` 中的量化/GEMM 接口，运行前需确认相应实现或环境存在。
- `07_MoE/`：稀疏 MoE、Top-k Router、Expert 和带 MoE 的小型语言模型实验。

## 08：注意力与推理加速

- `08_attention_mha_gqa_mla/`：MHA、MQA、GQA、MLA、Linear Attention 和教学版 Flash Attention 的独立实现，用于比较不同注意力机制的参数量、KV Cache 占用和计算方式。
- `08_kv_cache/`：KV Cache 原理与 GPT 变体实验，包括 MHA、GQA、MLA、MoE 和 FFN 相关缓存/性能对比。

## 09：解码与采样

`09_Sampling/` 实现 Temperature、Top-k、Top-p、组合采样和 Beam Search，重点理解 logits 变换、概率归一化、候选截断与随机性控制。

## A0：前沿模型训练与架构复现

`A0_frontier_model_training/` 用于研究较新的完整模型和训练工程：

- `qwen3/`：Qwen3 Dense/MoE 及 KV Cache 的独立 Notebook 实现。
- `qwen3.5/`：Qwen3.5 0.8B 的从零风格实现，包含线性注意力与全注意力交替结构以及 KV Cache 版本。
- `olmo3/`：OLMo 3 独立实现及 KV Cache 版本。
- `nanochat/`：较完整的预训练、SFT、评测和聊天工程，拥有独立 `pyproject.toml`，应视为子项目。
- `SmolLM3/`、`minimind-LLM/`：完整参考项目或训练资料，修改前先阅读各自 README、许可证和局部约定。

Qwen3、Qwen3.5、OLMo 3 的 `tests/` 会使用 tiny config 检查输出形状，并在依赖可用时把自实现权重映射到 Hugging Face 官方实现，使用 `torch.testing.assert_close` 验证数值一致性。这是前沿架构复现的优先验证方式。

## A1：预训练与监督微调

`A1_sft/` 包括：

- 预训练和全参数 SFT 脚本。
- 对话模板、数据预处理、Dataset、Collator 和 Trainer 流程。
- LoRA、QLoRA 的 PyTorch/NumPy 教学实现。
- DeepSpeed ZeRO 配置和训练 shell 脚本。
- 基于 Qwen 与 Alpaca 类数据的 Notebook 实验。

此模块可能依赖模型权重、在线数据集、GPU、多卡环境和实验跟踪服务，不应把完整训练当作默认验证方式。

## A2：偏好对齐与强化学习

`A2_rl-xpo/` 覆盖 RLHF 及多种策略优化方法：

- `dpo/`：Bradley-Terry 偏好模型、DPO 损失及完整数据到训练流程。
- `grpo/`：组内优势标准化、KL 估计与 GRPO 损失。
- `ppo/`：PPO/RLHF 推导和教学实现。
- `kto/`、`dapo/`、`reinforce++/`：其他偏好优化或在线强化学习实验。

分析此目录代码时，要明确区分 policy、reference model、reward/verifier、old policy、value model，以及 sequence-level reward 与 token-level log-probability。

## A3：评测

`A3_evaluatiom/`（目录名当前拼写如此）包括：

- 手写 BLEU、ROUGE-1、Perplexity 指标。
- LightEval、Ragas、DeepEval 和 LLM-as-a-Judge 的实验。

部分评测脚本可能需要外部模型、API 凭证或网络服务，运行前先检查环境，不要提交密钥。

# 参考

1. ~~[datawhalechina](https://github.com/datawhalechina)/[base-llm](https://github.com/datawhalechina/base-llm) （大模型基础与微调量化）~~
2. [datawhalechina](https://github.com/datawhalechina)/[happy-llm](https://github.com/datawhalechina/happy-llm)（从零构建大模型）
3. [datawhalechina](https://github.com/datawhalechina)/[diy-llm](https://github.com/datawhalechina/diy-llm)（斯坦福 CS336 中文课程）
4. [xiaodongguan/MA-RLHF](https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture)
5. [rasbt](https://github.com/rasbt)/[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) ✅
6. [The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)（HuggingFace Smol3模型训练手册）
7. [~~labml.ai 带注释的 PyTorch 版论文实现~~](https://nn.labml.ai/zh/index.html)
8. [deep-ml](https://www.deep-ml.com/collections) 类似于Leetcode一样的刷题网站
9. [duoan/TorchCode](https://github.com/duoan/TorchCode) 白板手撕 Transformer 面试的刷题平台
