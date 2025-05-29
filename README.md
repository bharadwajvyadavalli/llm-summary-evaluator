# LLM Evaluation System

A comprehensive system for evaluating LLM-generated content through two complementary approaches: **Summary Evaluation** (original POC) and **RAG Q&A Evaluation** (new system).

## 🎯 Overview

This project provides **dual evaluation capabilities**:

### 📝 **Summary Evaluation System (Original POC)**
- Evaluates LLM-generated summaries against reference text
- Uses mathematical metrics (ROUGE, BLEU, semantic similarity)
- LLM-as-judge evaluation for subjective criteria
- Quality categorization (Low/Medium/High) through clustering

### 🔍 **RAG Q&A Evaluation System (New)**
- Query any PDF documents with natural language questions
- Retrieval-Augmented Generation (RAG) with vector storage
- Evaluates response quality using RAG-specific metrics
- Works with any document type (no academic requirements)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM EVALUATION SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│  📝 SUMMARY EVALUATION        │  🔍 RAG Q&A EVALUATION     │
│                               │                             │
│  ┌─────────────────────────┐  │  ┌─────────────────────────┐ │
│  │ PDF Processing          │  │  │ PDF Processing          │ │
│  │ ↓                       │  │  │ ↓                       │ │
│  │ Mathematical Metrics    │  │  │ Vector Storage          │ │
│  │ (ROUGE, BLEU, etc.)     │  │  │ ↓                       │ │
│  │ ↓                       │  │  │ Query Processing        │ │
│  │ LLM Judge Evaluation    │  │  │ ↓                       │ │
│  │ ↓                       │  │  │ RAG Evaluation          │ │
│  │ Quality Clustering      │  │  │                         │ │
│  └─────────────────────────┘  │  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
llm-evaluation-system/
├── 📋 CONFIGURATION
│   ├── config.py                    # Shared configuration
│   ├── requirements.txt             # Dependencies
│   └── README.md                    # This file
│
├── 🎯 MAIN SYSTEMS
│   ├── main.py                      # Unified entry point (both systems)
│   │
│   ├── 📝 SUMMARY EVALUATION (Original POC)
│   ├── mathematical_metrics.py      # ROUGE, BLEU, semantic similarity
│   ├── llm_judge.py                 # LLM-as-judge evaluation
│   ├── clustering_model.py          # Quality categorization
│   │
│   └── 🔍 RAG Q&A EVALUATION (New)
│       ├── simple_pdf_processor.py  # PDF processing (any documents)
│       ├── vector_store_manager.py  # Vector storage & retrieval
│       ├── simple_rag_agent.py      # Query processing & response generation
│       └── rag_evaluator.py         # RAG-specific evaluation metrics
│
├── 🧪 TESTING & UTILITIES
│   ├── simple_test_system.py        # Comprehensive RAG testing
│   ├── quick_simple_test.py         # Quick RAG validation
│   └── verify_config.py             # Configuration verification
│
├── 📁 DATA & STORAGE
│   ├── sample_pdfs/                 # Input PDF documents
│   ├── vector_db/                   # Vector database (auto-created)
│   └── results/                     # Evaluation outputs (auto-created)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- PDF documents for evaluation

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-evaluation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# On Windows:
set OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Add Documents

```bash
# Create PDF directory
mkdir sample_pdfs

# Add your PDF files
cp your_documents/*.pdf sample_pdfs/
```

### 4. Quick Test

```bash
# Quick RAG system test
python quick_simple_test.py

# Verify configuration
python verify_config.py
```

## 📖 Usage Guide

### 🎯 Unified System (Both Evaluations)

Run both summary evaluation and RAG Q&A evaluation:

```bash
python main.py --pdfs sample_pdfs/ --output results/ --mode both
```

### 📝 Summary Evaluation Only

Evaluate LLM-generated summaries using mathematical metrics and LLM-as-judge:

```bash
python main.py --pdfs sample_pdfs/ --output results/ --mode summary
```

**Features:**
- ROUGE scores (1, 2, L)
- BLEU scores
- Semantic similarity
- LLM judge evaluation (relevance, coherence, fluency, factual accuracy, completeness)
- Quality clustering (Low/Medium/High)

### 🔍 RAG Q&A Evaluation Only

Query documents and evaluate response quality:

```bash
python main.py --pdfs sample_pdfs/ --output results/ --mode rag
```

**Features:**
- Vector-based document storage
- Natural language querying
- Context relevance evaluation
- Response faithfulness assessment
- Answer relevance scoring
- Completeness evaluation

### 🧪 Standalone RAG Testing

For detailed RAG system testing with custom queries:

```bash
# Comprehensive testing with customizable queries
python simple_test_system.py

# Quick validation test
python quick_simple_test.py
```

## ⚙️ Configuration Options

Edit `config.py` to customize:

### LLM Settings
```python
SUMMARY_MODEL = "gpt-3.5-turbo"    # Model for summary generation
JUDGE_MODEL = "gpt-4"              # Model for evaluation
```

### Summary Evaluation Weights
```python
JUDGE_CRITERIA = {
    "relevance": 0.25,
    "coherence": 0.20,
    "fluency": 0.15,
    "factual_accuracy": 0.25,
    "completeness": 0.15
}
```

### RAG Evaluation Weights
```python
RAG_EVALUATION_WEIGHTS = {
    "context_relevance": 0.25,
    "faithfulness": 0.30,
    "answer_relevance": 0.25,
    "completeness": 0.20
}
```

### Vector Store Settings
```python
CHUNK_SIZE = 1000           # Text chunk size for vector storage
CHUNK_OVERLAP = 200         # Overlap between chunks
SIMILARITY_SEARCH_K = 5     # Number of chunks to retrieve
```

## 📊 Output Files

### Summary Evaluation Results
- `summary_evaluation_results.csv` - Detailed metrics for all documents
- `summary_quality_categories.csv` - Quality classifications
- Mathematical metrics (ROUGE, BLEU, semantic similarity)
- LLM judge scores for each criterion
- Overall quality scores and categories

### RAG Evaluation Results
- `rag_evaluation_results.csv` - Query-response evaluation metrics
- Context relevance, faithfulness, answer relevance, completeness scores
- Source attribution and response metadata
- Performance analysis by query type

### Combined Report  
- `combined_report.html` - Executive summary with key insights

## 🎨 Customization Examples

### Custom Test Queries for RAG System

Edit `simple_test_system.py`:

```python
def get_sample_test_queries():
    return [
        # Business Documents
        "What are the key performance indicators mentioned?",
        "What are the main recommendations?",
        "What challenges are identified?",
        
        # Technical Documents  
        "What are the system requirements?",
        "How does the implementation work?",
        "What troubleshooting steps are provided?",
        
        # Research Documents
        "What methodology is used?",
        "What are the main findings?",
        "What are the limitations?",
        
        # Your Custom Queries
        "YOUR SPECIFIC QUESTION HERE"
    ]
```

### Custom Evaluation Criteria

For specialized domains, modify the evaluation criteria:

```python
# For technical documentation
JUDGE_CRITERIA = {
    "technical_accuracy": 0.30,
    "completeness": 0.25,
    "clarity": 0.20,
    "actionability": 0.25
}

# For business reports
RAG_EVALUATION_WEIGHTS = {
    "relevance_to_business": 0.30,
    "factual_accuracy": 0.35,
    "actionable_insights": 0.20,
    "completeness": 0.15
}
```

## 📈 Performance Benchmarks

### Expected Performance Ranges

**Summary Evaluation:**
- ROUGE-1: 0.3-0.7 (higher = better overlap)
- ROUGE-L: 0.2-0.6 (higher = better structure)
- BLEU: 0.1-0.4 (higher = better n-gram precision)
- LLM Judge: 5-9/10 (higher = better quality)

**RAG Evaluation:**
- Context Relevance: 0.6-0.9 (higher = better retrieval)
- Faithfulness: 0.7-0.95 (higher = more accurate)
- Answer Relevance: 0.65-0.85 (higher = more relevant)
- Overall RAG Score: 0.65-0.85 (higher = better quality)

### Quality Thresholds
- **Excellent**: RAG Score ≥ 0.8
- **Good**: RAG Score 0.65-0.8
- **Fair**: RAG Score 0.5-0.65
- **Poor**: RAG Score < 0.5

## 🔧 Troubleshooting

### Common Issues

**1. Config Attribute Error**
```bash
AttributeError: module 'config' has no attribute 'JUDGE_CRITERIA'
```
**Solution:** Run `python verify_config.py` to check configuration

**2. OpenAI API Error**
```bash
Error: Invalid API key
```
**Solution:** Verify API key is set correctly:
```bash
echo $OPENAI_API_KEY  # Should show your key
```

**3. No Documents Processed**
```bash
❌ No documents were processed
```
**Solution:** 
- Check PDF files are in `sample_pdfs/` directory
- Verify PDFs contain extractable text (not scanned images)
- Run `python verify_config.py` for detailed diagnostics

**4. Vector Database Issues**
```bash
Collection [documents] does not exist
```
**Solution:** Delete and recreate vector database:
```bash
rm -rf vector_db/
python quick_simple_test.py
```

**5. Memory Issues with Large PDFs**
```bash
Out of memory error
```
**Solution:** Reduce chunk size in `config.py`:
```python
CHUNK_SIZE = 500
SIMILARITY_SEARCH_K = 3
```

### Debug Commands

```bash
# Check system configuration
python verify_config.py

# Test PDF processing only
python -c "
from simple_pdf_processor import SimplePDFProcessor
processor = SimplePDFProcessor()
docs = processor.process_pdfs('sample_pdfs')
print(f'Processed {len(docs)} documents')
"

# Test vector storage
python -c "
from vector_store_manager import VectorStoreManager
vm = VectorStoreManager()
stats = vm.get_collection_stats()
print(f'Vector store: {stats}')
"
```

## 🔬 Advanced Usage

### Batch Processing Multiple Directories

```python
import os
from pathlib import Path

# Process multiple PDF directories
pdf_directories = ["docs1/", "docs2/", "docs3/"]

for pdf_dir in pdf_directories:
    if os.path.exists(pdf_dir):
        os.system(f"python main.py --pdfs {pdf_dir} --output results_{Path(pdf_dir).name} --mode both")
```

### Custom Evaluation Pipeline

```python
from simple_pdf_processor import SimplePDFProcessor
from vector_store_manager import VectorStoreManager
from simple_rag_agent import SimpleRAGAgent
from rag_evaluator import RAGEvaluator

# Custom processing pipeline
processor = SimplePDFProcessor()
documents = processor.process_pdfs("sample_pdfs")

# Custom queries based on document type
if "technical" in documents[0]['text'].lower():
    queries = ["What are the technical specifications?", "How does it work?"]
elif "business" in documents[0]['text'].lower():
    queries = ["What are the KPIs?", "What are the recommendations?"]
else:
    queries = ["What is the main content?", "What are key points?"]

# Process and evaluate
vector_manager = VectorStoreManager()
vector_manager.add_documents(documents)
agent = SimpleRAGAgent(vector_manager)
evaluator = RAGEvaluator()

for query in queries:
    response = agent.query(query)
    metrics = evaluator.evaluate_response(response)
    print(f"Query: {query}")
    print(f"Score: {metrics['rag_score']:.3f}")
```

## 📚 API Reference

### SimplePDFProcessor
```python
processor = SimplePDFProcessor()
documents = processor.process_pdfs(pdf_directory)
```

### SimpleRAGAgent
```python
agent = SimpleRAGAgent(vector_manager)
response = agent.query(question, k=5)  # k = number of chunks to retrieve
```

### RAGEvaluator
```python
evaluator = RAGEvaluator()
metrics = evaluator.evaluate_response(query_response)
# Returns: context_relevance, faithfulness, answer_relevance, completeness, rag_score
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- ROUGE and BLEU evaluation metrics
- PyPDF2 for PDF processing

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Run `python verify_config.py` for diagnostics
3. Create an issue in the repository

---

**Version:** 2.0  
**Last Updated:** 2024  
**Compatibility:** Python 3.8+, OpenAI API v1.0+