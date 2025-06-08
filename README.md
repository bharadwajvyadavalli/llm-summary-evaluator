# LLM Evaluation System

A streamlined system for evaluating LLM-generated document summaries and question-answering capabilities using mathematical metrics, LLM judge evaluation, and quality categorization.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Usage Modes

#### 1. Training Mode
Train a quality assessment model from PDF documents:

```bash
python main.py train --input pdfs/training/ --output output/
```

This will:
- Process PDFs and extract abstracts
- **Generate 3 different quality summaries per document (High/Medium/Low)**
- Calculate metrics (ROUGE, semantic similarity) for each summary
- Train unsupervised clustering model on diverse quality examples
- Save model as `output/quality_model.pkl`
- Generate training report showing cluster alignment

#### 2. Inference Mode
Evaluate new PDFs using trained model:

```bash
python main.py inference --input pdfs/test/ --model output/quality_model.pkl --output results/
```

#### 3. Query Mode
Answer questions using vector search and evaluate response quality:

```bash
# Single question
python main.py query --pdfs pdfs/docs/ --model output/quality_model.pkl --queries "What is machine learning?" --output results/

# Multiple questions from CSV
python main.py query --pdfs pdfs/docs/ --model output/quality_model.pkl --queries questions.csv --output results/
```

## 📁 Project Structure

```
├── main.py           # Entry point with train/inference/query modes
├── processor.py      # PDF processing & summary generation
├── metrics_evals.py  # ROUGE metrics & LLM judge evaluation
├── model.py          # Quality assessment model
├── vector_index.py   # Vector search for query evaluation
├── config.py         # Configuration settings
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 📊 Features

### PDF Processing & Storage
- Extracts text and abstracts from PDFs
- Generates summaries using configurable prompts
- Stores data as CSV (text, abstract, summary)

### Evaluation & Categorization
- Computes ROUGE scores, semantic similarity
- LLM judge evaluation (relevance, coherence, completeness)
- Unsupervised clustering into High/Medium/Low quality
- Saves trained model as .pkl file

### Question Answering & Scoring
- Creates vector indexes from PDFs
- Retrieves relevant chunks for questions
- Generates answers using LLM
- Evaluates answer quality
- Stores all outputs (question, answer, chunks, metrics, quality)

### Reporting & Export
- Professional HTML reports with statistics
- CSV exports for further analysis
- Quality distribution visualizations

## 📝 Input/Output

### Training Input
- Directory of PDF research papers
- Papers should have extractable abstracts

### Training Output
- `quality_model.pkl` - Trained model
- `training_report.html` - Summary report with:
  - Distribution of generated quality levels
  - Confusion matrix showing model clustering vs intended quality
  - Performance metrics

### Query Output
- `query_results.csv` - Q&A evaluations
- `query_report.html` - Interactive report

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Models
SUMMARY_MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4"

# Quality-specific prompts for training
SUMMARY_PROMPTS = {
    "high": "Comprehensive, accurate summary...",
    "medium": "Clear summary with basic coverage...",
    "low": "Brief, simple summary..."
}

# Evaluation criteria
JUDGE_CRITERIA = ['relevance', 'coherence', 'completeness']

# Clustering
N_CLUSTERS = 3  # Low, Medium, High
```

### Training Strategy

The system generates **3 different quality summaries** per document during training:

1. **High Quality**: Comprehensive, detailed, academically rigorous
2. **Medium Quality**: Good coverage but may omit technical details  
3. **Low Quality**: Brief, basic, may miss key points

This diversity helps the unsupervised clustering model learn to distinguish quality levels more effectively.

## 📈 Example Workflow

```bash
# 1. Prepare training data
mkdir -p pdfs/training
# Add 50-60 research PDFs

# 2. Train model
python main.py train --input pdfs/training/ --output model/

# 3. Prepare test questions
echo "question" > questions.csv
echo "What are the main applications of deep learning?" >> questions.csv
echo "How does transfer learning work?" >> questions.csv

# 4. Index documents for Q&A
mkdir -p pdfs/knowledge_base
# Add relevant PDFs

# 5. Run Q&A evaluation
python main.py query --pdfs pdfs/knowledge_base/ --model model/quality_model.pkl --queries questions.csv --output qa_results/

# 6. View results
open qa_results/query_report.html
```

## 🔧 Advanced Usage

### Custom Evaluation Metrics

Add new metrics in `metrics_evals.py`:

```python
def custom_metric(self, text, reference):
    # Your implementation
    return score
```

### Batch Processing

Process large datasets:

```bash
for dir in pdfs/batch_*; do
    python main.py inference --input "$dir" --model model/quality_model.pkl --output "results/$(basename $dir)"
done
```

## 📊 Output Examples

### Quality Distribution
- **High Quality**: Strong alignment with reference, coherent, complete
- **Medium Quality**: Good coverage, minor issues
- **Low Quality**: Poor relevance, missing key information

### Metrics Included
- ROUGE-1, ROUGE-2, ROUGE-L scores
- Semantic similarity (cosine similarity of embeddings)
- Compression ratio
- LLM judge scores (relevance, coherence, completeness)
- Overall quality score (weighted average)

## 🚨 Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `BATCH_SIZE` in config.py
2. **Memory Issues**: Process PDFs in smaller batches
3. **PDF Extraction Errors**: Ensure PDFs are text-based, not scanned images

### Debug Mode

Enable verbose logging:
```python
# In any script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License

MIT License - Feel free to adapt for your needs!