
from lighteval.pipeline import Pipeline

"""
LightEval is the toolkit used by the Hugging Face Leaderboard team to evaluate models on standard benchmarks like MMLU or GSM8K. 
"""

# Simplified example of running a benchmark task
pipeline = Pipeline(
    model="gpt2", # Or any HF model path
    tasks="mmlu", 
    device="cuda"
)
results = pipeline.evaluate()
print(results)
