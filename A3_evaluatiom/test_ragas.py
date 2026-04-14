from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)

"""
RAGAS 是目前 RAG 评估领域最主流的开源框架，它的核心设计理念是把 RAG 系统拆成检索和生成两个环节分别评估，这样能精准定位瓶颈所在，而不是只看端到端结果。

它定义了四个核心指标：
- 检索端有 Context Precision 衡量检索结果中有多少是真正有用的、Context Recall 衡量回答所需的关键信息有没有被检索到；
- 生成端有 Faithfulness 检查答案是否忠实于检索到的上下文、没有出现幻觉，Answer Relevancy 检查答案是否紧扣用户问题、没有跑题。
这四个指标正好形成了一个"精确性×完整性"和"检索×生成"的评估矩阵。

评估流程上，RAGAS 基于 LLM-as-Judge 实现自动化评估。
每条评估样本需要准备 question、contexts、answer 和 ground truth 四个字段，框架会用 LLM 逐指标打分，
比如 Faithfulness 会先把答案拆解成独立陈述再逐一验证是否有上下文支持。
拿到各指标得分后做交叉分析——如果 Context Recall 高但 Faithfulness 低，说明检索没问题但生成有幻觉；反过来则说明要优化检索策略。

评估数据的获取在实际项目中是最费力的环节。我们通常三种方式组合：
1. 项目早期用 RAGAS 自带的合成数据生成功能基于知识库自动构造 Q&A 对来快速建基线；
2. 系统上线后从生产日志里提取真实用户查询、结合隐式反馈筛选困难样本；再由领域专家做质量审核和边界 case 补充。
"""

# Example data
data = {
    "query": ["What is the capital of France?"],
    "generated_response": ["Paris is the capital of France."],
    "retrieved_documents": [["Paris is the capital of France. It is a major European city known for its culture."]]
}

# Convert the data to a Hugging Face Dataset
dataset = Dataset.from_dict(data)

# Define the metrics you want to evaluate
metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
]

# Evaluate the dataset using the selected metrics
results = evaluate(dataset, metrics)

# Display the results
for metric_name, score in results.items():
    print(f"{metric_name}: {score:.2f}")