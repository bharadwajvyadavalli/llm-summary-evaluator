"""
Configuration settings for LLM Evaluation Service
"""

import os

# API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"

# Summary Generation Settings
SUMMARY_LEVELS = {
    'high': {
        'prompt': "Summarize in 1-2 sentences:",
        'max_tokens': 50,
        'input_chars': 2000
    },
    'medium': {
        'prompt': "Summarize in one paragraph:",
        'max_tokens': 150,
        'input_chars': 5000
    },
    'low': {
        'prompt': "Provide a detailed multi-paragraph summary:",
        'max_tokens': 500,
        'input_chars': 10000
    }
}

# Model Settings
N_CLUSTERS = 3  # Low, Medium, High quality
QUALITY_LABELS = ["Low", "Medium", "High"]

# Vector Index Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SEARCH_K = 3  # Number of references to retrieve

# Processing Settings
PDF_TIMEOUT = 30  # Seconds for PDF download
BATCH_SIZE = 10
TEMPERATURE = 0.3  # For consistent summaries

# LLM Judge Settings
JUDGE_MODEL = "gpt-4"  # Use GPT-4 for better evaluation
JUDGE_CRITERIA = {
    "relevance": {
        "weight": 0.25,
        "prompt": "Rate the RELEVANCE of this summary compared to the abstract on a scale of 1-10. Consider how well the summary captures the main points and findings."
    },
    "coherence": {
        "weight": 0.20,
        "prompt": "Rate the COHERENCE and logical flow of this summary on a scale of 1-10. Consider the structure, transitions, and logical progression."
    },
    "fluency": {
        "weight": 0.15,
        "prompt": "Rate the FLUENCY and language quality of this summary on a scale of 1-10. Consider grammar, clarity, and readability."
    },
    "factual_accuracy": {
        "weight": 0.25,
        "prompt": "Rate the FACTUAL ACCURACY of this summary compared to the abstract on a scale of 1-10. Check for any errors or misrepresentations."
    },
    "completeness": {
        "weight": 0.15,
        "prompt": "Rate the COMPLETENESS of this summary compared to the abstract on a scale of 1-10. Consider whether key findings and conclusions are included."
    }
}

# Which summary level to evaluate with LLM judge
JUDGE_SUMMARY_LEVEL = "medium"  # Best balance for evaluation