"""Configuration Settings"""

import os

# API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Settings
SUMMARY_MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4"
ANSWER_MODEL = "gpt-3.5-turbo"

# Summary Generation
MAX_SUMMARY_LENGTH = 400

# Quality-specific prompts for training diversity
SUMMARY_PROMPTS = {
    "high": """You are an expert research summarizer. Create a comprehensive, accurate summary that captures:
1. Main research question/objective with context
2. Detailed methodology and approach
3. All primary findings and key results
4. Main conclusions and implications
5. Limitations and future work

Maintain academic rigor, use precise terminology, and ensure all key points are covered. Length: 250-300 words.""",

    "medium": """Summarize this research paper covering:
- The main research goal
- Basic methodology used
- Key findings
- Main conclusions

Be clear but you may omit some technical details. Length: 150-200 words.""",

    "low": """Briefly summarize this paper in simple terms. Just mention what the research is about and the main finding. 
Keep it very short and basic. Length: 50-100 words. You can skip technical details and methodology."""
}

# Evaluation Criteria
JUDGE_CRITERIA = [
    'relevance',
    'coherence',
    'completeness'
]

# Clustering Settings
N_CLUSTERS = 3
QUALITY_LABELS = ["Low", "Medium", "High"]

# Vector Search Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 5

# Processing Settings
BATCH_SIZE = 10