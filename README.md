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
Create vector indexes from new PDFs:

```bash
python main.py inference --input pdfs/test/ --output indexes/
```

This will:
- Process PDFs and extract text
- Create chunks with overlap for better retrieval
- Generate embeddings for all chunks
- Save index statistics and chunk details
- Generate index report showing document coverage

#### 3. Query Mode
Answer questions using vector search and evaluate response quality:

```bash
# Using questions from config.py (recommended for testing)
python main.py query --index indexes/vector_index/ --model output/quality_model.pkl --queries config --output results/

# Using questions from CSV file
python main.py query --index indexes/vector_index/ --model output/quality_model.pkl --queries questions.csv --output results/

# Using a single question
python main.py query --index indexes/vector_index/ --model output/quality_model.pkl --queries "What is machine learning?" --output results/

# Or create index on-the-fly from PDFs
python main.py query --pdfs pdfs/docs/ --model output/quality_model.pkl --queries config --output results/
```

#### 4. Report Mode
Generate professional reports from query results:

```bash
python main.py report --input results/query_results.csv --output reports/
```

This will generate:
- `professional_report.html` - Interactive executive report with charts and insights
- `metrics_summary.csv` - Statistical summary of all metrics
- `quality_breakdown.csv` - Quality distribution analysis
- `question_performance.csv` - Detailed per-question performance

```
├── main.py           # Entry point with train/inference/query modes
├── processor.py      # PDF processing & summary generation
├── metrics_evals.py  # ROUGE metrics & LLM judge evaluation
├── models.py         # Quality assessment model
├── vector_index.py   # Vector search for query evaluation
├── config.py         # Configuration settings
├── test_system.py    # Testing script
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

### Query Output
- `query_results.csv` - Q&A evaluations
- `query_report.html` - Interactive report

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

### Inference Output
- `vector_index/` - Directory containing saved vector index:
  - `chunks.json` - All text chunks with metadata
  - `embeddings.npy` - NumPy array of chunk embeddings
  - `metadata.json` - Index metadata
- `index_stats.csv` - Vector index statistics
- `chunks_index.csv` - Details of all chunks created
- `index_report.html` - Summary report

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

# 3. Prepare documents for indexing
mkdir -p pdfs/knowledge_base
# Add relevant PDFs

# 4. Create vector indexes
python main.py inference --input pdfs/knowledge_base/ --output indexes/

# 5. Prepare test questions
echo "question" > questions.csv
echo "What are the main applications of deep learning?" >> questions.csv
echo "How does transfer learning work?" >> questions.csv

# 6. Run Q&A evaluation using pre-built index
python main.py query --index indexes/vector_index/ --model model/quality_model.pkl --queries questions.csv --output qa_results/

# 7. Generate professional reports
python main.py report --input qa_results/query_results.csv --output qa_reports/

# 8. View results
open qa_reports/professional_report.html
```

## 🔧 Advanced Usage

### Professional Reporting

Generate comprehensive reports from query results:

```bash
# Run queries and generate initial results
python main.py query --index indexes/vector_index/ --model model/quality_model.pkl --queries config --output results/

# Generate professional reports
python main.py report --input results/query_results.csv --output reports/

# Reports include:
# - Executive summary with key metrics
# - Interactive charts and visualizations
# - Performance distribution analysis
# - Question-level breakdown
# - Correlation analysis
```

### Using Pre-built Indexes

Save time by reusing vector indexes:

```bash
# Build index once
python main.py inference --input pdfs/corpus/ --output indexes/

# Use the same index for multiple query sessions
python main.py query --index indexes/vector_index/ --model model/quality_model.pkl --queries batch1.csv --output results1/
python main.py query --index indexes/vector_index/ --model model/quality_model.pkl --queries batch2.csv --output results2/
```

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
    python main.py inference --input "$dir" --output "indexes/$(basename $dir)"
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