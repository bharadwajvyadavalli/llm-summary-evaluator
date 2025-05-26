# Quick Setup Guide - LLM Evaluation Service POC

## Files to Download/Create

Save each file in the same directory:

1. **README.md** - Project documentation
2. **requirements.txt** - Python dependencies  
3. **config.py** - Configuration settings
4. **main.py** - Main script to run
5. **document_processor.py** - PDF processing and summary generation
6. **mathematical_metrics.py** - ROUGE, BLEU, semantic similarity
7. **llm_judge.py** - LLM-as-judge evaluation
8. **clustering_model.py** - Quality categorization
9. **sample_pdfs/** - Sample PDF directory (create and add PDFs)

## Quick Start

1. **Create project folder:**
```bash
mkdir llm-evaluation-poc
cd llm-evaluation-poc
```

2. **Save all the files above to this folder**

3. **Set up API key:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Create sample PDF directory:**
```bash
mkdir sample_pdfs
# Add your PDF files here
```

6. **Run the evaluation:**
```bash
python main.py --pdfs sample_pdfs/ --output results/
```

7. **View results:**
- `results/evaluation_results.csv` - All metrics
- `results/quality_categories.csv` - Final classifications
- `results/summary_report.html` - Executive summary

## Using Your Own Data

- Add 10-60 PDF research papers to `sample_pdfs/`
- Papers should have clear abstracts (used as golden records)
- System will extract abstracts automatically and generate summaries

## Customization

Edit `config.py` to:
- Change LLM models
- Adjust evaluation criteria weights
- Modify clustering parameters

## What It Does

1. **Extracts** abstracts from PDFs (golden records)
2. **Generates** summaries using LLM
3. **Calculates** mathematical metrics (ROUGE, BLEU, similarity)
4. **Evaluates** subjective criteria using LLM judge
5. **Categorizes** into Low/Medium/High quality using clustering

Simple POC ready to run! 🚀