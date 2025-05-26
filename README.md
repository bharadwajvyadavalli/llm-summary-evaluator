# LLM Evaluation Service POC

Simple proof-of-concept for evaluating LLM-generated document summaries using mathematical metrics, LLM-as-judge, and quality categorization.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. Run evaluation:
```bash
python main.py --pdfs sample_pdfs/ --output results/
```

## How It Works

1. **Process PDFs** - Extract text and abstracts (golden records)
2. **Calculate Mathematical Metrics** - ROUGE, BLEU, semantic similarity
3. **LLM Judge Evaluation** - Relevance, coherence, accuracy, completeness  
4. **Quality Categorization** - Unsupervised clustering → Low/Medium/High

## Input

- PDF documents (50-60 recommended)
- Abstracts extracted automatically as reference
- Summaries generated or provided

## Output

- `results/evaluation_results.csv` - All metrics
- `results/quality_categories.csv` - Final classifications
- `results/summary_report.html` - Executive report

## Configuration

Edit `config.py` to customize:
- LLM provider (OpenAI/Anthropic)
- Evaluation criteria weights
- Clustering parameters

Simple POC - extend as needed for production use.