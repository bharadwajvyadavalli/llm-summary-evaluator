# LLM Evaluation Service

A streamlined service for evaluating LLM-generated summaries with quality assessment and vector-based query evaluation.

## 📁 Project Structure

```
├── main.py           # Entry point (train/inference/vector-query modes)
├── processor.py      # PDF processing & summary generation
├── metrics_evals.py  # ROUGE metrics & LLM judge evaluation
├── model.py          # Quality assessment model
├── vector_index.py   # Vector search for query evaluation
├── config.py         # Configuration settings
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 🚀 Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## 📋 Requirements

```
pandas
numpy
scikit-learn
PyPDF2
openai
rouge-score
sentence-transformers
faiss-cpu
requests
```

## 🔍 Testing Vector Index

**Simple Testing with Saved Index:**
```bash
# First time - builds and saves index
python main.py --mode vector-query_test --vector-db-dir my_index --index-dir pdfs/ --query "test" --model model.pkl

# Subsequent times - reuses saved index
python main.py --mode vector-query_test --vector-db-dir my_index --query "new query" --model model.pkl

# If index is corrupted, rebuild:
rm -rf my_index/
python main.py --mode vector-query_test --vector-db-dir my_index --index-dir pdfs/ --query "test" --model model.pkl
```

## 🎯 Usage

### 1. Training Mode (Steps 1-4)
Train a quality assessment model on your PDF documents:

```bash
# Using local PDFs
python main.py --mode train --source /path/to/pdfs/ --model quality_model.pkl

# Using URLs
python main.py --mode train --source "https://url1.pdf,https://url2.pdf" --model quality_model.pkl
```

**What it does:**
- Downloads/loads PDFs
- Generates 3-level summaries (high/medium/low)
- Computes mathematical metrics (ROUGE, semantic similarity)
- Evaluates with LLM judge (relevance, coherence, accuracy, etc.)
- Trains unsupervised model
- Saves model to .pkl file

### 2. Inference Mode (Step 5)
Evaluate a single document:

```bash
python main.py --mode inference --document test.pdf --model quality_model.pkl
```

**Output:**
```
Document: test.pdf
Quality Score: 7.8/10
Confidence: 89%
Summaries:
  High-level: [1-2 sentence summary]
  Medium-level: [paragraph summary]
Key Metrics:
  Mathematical:
    avg_rouge: 0.823
    avg_semantic: 0.756
    medium_rougeL: 0.812
  LLM Judge:
    relevance: 8.5/10
    coherence: 7.8/10
    accuracy: 8.2/10
    Overall: 8.1/10
```

### 3. Vector Query Mode (Step 6)
Evaluate LLM responses using reference documents:

```bash
# Standard mode (rebuilds index each time)
python main.py --mode vector-query \
  --index-dir reference_docs/ \
  --query "What is attention mechanism?" \
  --model quality_model.pkl

# Test mode (saves/reuses index)
python main.py --mode vector-query_test \
  --vector-db-dir vector_db \
  --index-dir sample_pdfs/ \
  --query "What is attention mechanism?" \
  --model quality_model.pkl

# Subsequent runs (uses saved index)
python main.py --mode vector-query_test \
  --vector-db-dir vector_db \
  --query "Explain transformers" \
  --model quality_model.pkl
```

**Output:**
```
📂 Loading existing index from vector_db/vector_index.pkl
Query: What is attention mechanism?
LLM Response: [generated response]
Found 3 relevant references

📊 Results:
Quality Score: 8.2/10
1. attention_paper.pdf (similarity: 0.892)
2. transformer_guide.pdf (similarity: 0.834)
3. bert_explained.pdf (similarity: 0.756)
```

## 🧪 Testing

### Quick Test Without Real PDFs

```python
# test_example.py
import numpy as np
import pandas as pd
from model import QualityModel

# Create synthetic data
data = {
    'document': ['doc1.pdf', 'doc2.pdf', 'doc3.pdf'],
    'high_rouge1': [0.8, 0.6, 0.4],
    'medium_rouge1': [0.85, 0.65, 0.45],
    'low_rouge1': [0.9, 0.7, 0.5],
    'avg_rouge': [0.85, 0.65, 0.45],
    'avg_semantic': [0.9, 0.7, 0.5]
}

df = pd.DataFrame(data)

# Train model
model = QualityModel()
model.train(df)
model.save('test_model.pkl')

# Test prediction
test_metrics = {
    'high_rouge1': 0.75,
    'medium_rouge1': 0.8,
    'low_rouge1': 0.85,
    'avg_rouge': 0.8,
    'avg_semantic': 0.82
}

score, confidence = model.predict(test_metrics)
print(f"Quality Score: {score:.1f}/10, Confidence: {confidence:.1%}")
```

### Sample URLs for Testing

```bash
# arXiv papers (computer science)
python main.py --mode train --source \
"https://arxiv.org/pdf/2005.14165.pdf,\
https://arxiv.org/pdf/1706.03762.pdf,\
https://arxiv.org/pdf/1810.04805.pdf" \
--model arxiv_model.pkl
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **OpenAI settings**: Model name, temperature
- **Summary levels**: Prompts and token limits
- **Clustering**: Number of quality categories
- **Vector search**: Embedding model, number of results

### Key Settings:

```python
# config.py
OPENAI_MODEL = "gpt-3.5-turbo"  # For summaries
JUDGE_MODEL = "gpt-4"  # For evaluation
N_CLUSTERS = 3  # Quality levels: Low/Medium/High
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer

# LLM Judge criteria (with weights)
JUDGE_CRITERIA = {
    "relevance": {"weight": 0.25, ...},
    "coherence": {"weight": 0.20, ...},
    "fluency": {"weight": 0.15, ...},
    "factual_accuracy": {"weight": 0.25, ...},
    "completeness": {"weight": 0.15, ...}
}
```

## 📊 Understanding the Output

### Quality Scores (0-10)
- **8-10**: High quality (accurate, coherent, complete)
- **5-7**: Medium quality (good but with minor issues)
- **0-4**: Low quality (significant problems)

### Confidence Scores
- Based on distance to cluster centers
- Higher confidence = more typical of the quality category

### Metrics Explained
- **ROUGE**: Word overlap with reference text
- **Semantic Similarity**: Meaning preservation (0-1)
- **Compression Ratio**: Summary length vs original
- **LLM Judge Scores** (1-10):
  - **Relevance**: How well summary captures main points
  - **Coherence**: Logical flow and structure
  - **Fluency**: Language quality and readability
  - **Factual Accuracy**: Correctness vs reference
  - **Completeness**: Coverage of key information

## 🔧 Troubleshooting

### Vector Index Error Fix
If you see: `ValueError: The truth value of an array...`
```bash
# Quick fix:
rm -rf vector_db/
python main.py --mode vector-query_test --vector-db-dir vector_db --index-dir your_pdfs/ --query "test" --model quality_model.pkl
```

### Common Issues

1. **No OpenAI API key**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **PDF extraction errors**
   - Ensure PDFs are text-based (not scanned images)
   - Try different PDFs or use the URL download option

3. **Out of memory**
   - Process fewer PDFs at once
   - Reduce batch size in config.py

4. **Import errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

5. **Vector index errors**
   ```bash
   # Check if index is corrupted
   python fix_vector_index.py vector_db/vector_index.pkl
   
   # Rebuild if needed
   python fix_vector_index.py vector_db/vector_index.pkl your_pdfs/
   ```

## 📈 Extending the System

### Test Multiple Queries
```bash
# Use test_vector_simple.py
python test_vector_simple.py

# Or use the shell script
./test_vector.sh
```

### Add New Metrics
Edit `metrics.py`:
```python
def compute_custom_metric(self, summary, reference):
    # Your metric logic here
    return score
```

### Change Quality Categories
Edit `config.py`:
```python
N_CLUSTERS = 5  # More granular: Very Low/Low/Medium/High/Very High
QUALITY_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]
```

### Custom Summary Prompts
Edit `config.py`:
```python
SUMMARY_LEVELS = {
    'executive': {
        'prompt': "Write an executive summary:",
        'max_tokens': 200,
        'input_chars': 8000
    }
}
```

## 🎉 Example Workflow

```bash
# 1. Train model
python main.py --mode train --source my_papers/ --model my_model.pkl

# 2. Test on single document
python main.py --mode inference --document new_paper.pdf --model my_model.pkl

# 3. Vector search - first time (builds index)
python main.py --mode vector-query_test \
  --vector-db-dir my_vector_db \
  --index-dir reference_docs/ \
  --query "What is BERT?" \
  --model my_model.pkl

# 4. Vector search - subsequent times (reuses index)
python main.py --mode vector-query_test \
  --vector-db-dir my_vector_db \
  --query "Explain attention mechanism" \
  --model my_model.pkl
```

## 📝 License

MIT License - Feel free to modify and use!