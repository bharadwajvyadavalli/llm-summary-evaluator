# Complete Configuration - Works with both Original POC and New RAG System

import os

# LLM Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUMMARY_MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4"

# Vector Store Settings (for RAG system)
VECTOR_DB_PATH = "./vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SIMILARITY_SEARCH_K = 5

# Original POC - Evaluation Criteria Weights (for llm_judge.py)
JUDGE_CRITERIA = {
    "relevance": 0.25,
    "coherence": 0.20,
    "fluency": 0.15,
    "factual_accuracy": 0.25,
    "completeness": 0.15
}

# New RAG System - Evaluation Weights (for rag_evaluator.py)
RAG_EVALUATION_WEIGHTS = {
    "context_relevance": 0.25,
    "faithfulness": 0.30,
    "answer_relevance": 0.25,
    "completeness": 0.20
}

# Original POC - Clustering Settings (for clustering_model.py)
N_CLUSTERS = 3
CLUSTER_LABELS = ["Low", "Medium", "High"]

# Processing Settings
MAX_SUMMARY_LENGTH = 300
BATCH_SIZE = 5