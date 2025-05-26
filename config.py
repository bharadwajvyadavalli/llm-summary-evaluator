# Configuration for LLM Evaluation Service POC

import os

# LLM Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUMMARY_MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4"

# Evaluation Criteria Weights
JUDGE_CRITERIA = {
    "relevance": 0.25,
    "coherence": 0.20, 
    "fluency": 0.15,
    "factual_accuracy": 0.25,
    "completeness": 0.15
}

# Clustering Settings
N_CLUSTERS = 3  # Low, Medium, High
CLUSTER_LABELS = ["Low", "Medium", "High"]

# Processing Settings
MAX_SUMMARY_LENGTH = 500
BATCH_SIZE = 5