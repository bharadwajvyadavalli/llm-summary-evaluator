# LLM Evaluation System - Complete Setup Guide

This guide will walk you through setting up the LLM Evaluation System from scratch, with detailed steps for different operating systems and use cases.

## 📋 Table of Contents

1. [System Requirements](#-system-requirements)
2. [Installation Methods](#-installation-methods)
3. [Step-by-Step Setup](#-step-by-step-setup)
4. [Configuration](#-configuration)
5. [First Run](#-first-run)
6. [Validation & Testing](#-validation--testing)
7. [Common Issues](#-common-issues)
8. [Advanced Setup](#-advanced-setup)

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space (more for large document collections)
- **Internet**: Required for OpenAI API and package installation

### Recommended Requirements
- **RAM**: 16GB for processing large document collections
- **Storage**: 10GB+ for extensive document libraries
- **CPU**: Multi-core processor for faster processing

### Required Accounts
- **OpenAI API Account** with available credits
  - Sign up at [platform.openai.com](https://platform.openai.com)
  - Minimum $5 in credits recommended

## 🛠️ Installation Methods

Choose your preferred installation method:

### Method 1: Fresh Installation (Recommended)
Complete setup from scratch with virtual environment

### Method 2: Existing Project Update
Update an existing project with new components

### Method 3: Docker Installation
Containerized setup (advanced users)

---

## 🚀 Step-by-Step Setup

### Method 1: Fresh Installation

#### Step 1: System Preparation

**Windows:**
```cmd
# Check Python version
python --version
# Should show Python 3.8+

# Install pip if missing
python -m ensurepip --upgrade
```

**macOS:**
```bash
# Check Python version
python3 --version

# Install Python if missing (using Homebrew)
brew install python3

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
# Update system
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Check version
python3 --version
```

#### Step 2: Project Setup

```bash
# Create project directory
mkdir llm-evaluation-system
cd llm-evaluation-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

#### Step 3: Install Dependencies

```bash
# Create requirements.txt
cat > requirements.txt << EOF
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
PyPDF2>=3.0.0
openai>=1.0.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
rouge-score>=0.1.2
nltk>=3.8.0
EOF

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(openai|chromadb|sentence-transformers)"
```

#### Step 4: Download Project Files

**Option A: Download Files Manually**
Create each file from the project structure with the provided code.

**Option B: Clone Repository (if available)**
```bash
git clone <repository-url> .
```

**Option C: Create Files Manually**
```bash
# Create all required files (see file list below)
touch config.py main.py simple_pdf_processor.py
touch simple_rag_agent.py vector_store_manager.py rag_evaluator.py
touch mathematical_metrics.py llm_judge.py clustering_model.py
touch simple_test_system.py quick_simple_test.py verify_config.py
```

#### Step 5: File Structure Verification

Your directory should look like this:
```
llm-evaluation-system/
├── venv/                          # Virtual environment
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── main.py                        # Main entry point
├── simple_pdf_processor.py        # PDF processing
├── simple_rag_agent.py           # RAG query processing
├── vector_store_manager.py       # Vector storage
├── rag_evaluator.py              # RAG evaluation
├── mathematical_metrics.py       # Original metrics
├── llm_judge.py                  # Original LLM judge
├── clustering_model.py           # Original clustering
├── simple_test_system.py         # RAG testing
├── quick_simple_test.py          # Quick test
└── verify_config.py              # Config verification
```

---

## ⚙️ Configuration

### Step 1: OpenAI API Setup

1. **Get OpenAI API Key:**
   - Visit [platform.openai.com](https://platform.openai.com)
   - Create account or log in
   - Go to API Keys section
   - Create new secret key
   - Copy the key (starts with `sk-...`)

2. **Set Environment Variable:**

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-your-actual-key-here
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-actual-key-here"
```

**macOS/Linux (Bash):**
```bash
export OPENAI_API_KEY="sk-your-actual-key-here"

# Make it permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-your-actual-key-here"' >> ~/.bashrc
source ~/.bashrc
```

3. **Verify API Key:**
```bash
# Check if set correctly
echo $OPENAI_API_KEY  # macOS/Linux
echo %OPENAI_API_KEY%  # Windows CMD
echo $env:OPENAI_API_KEY  # Windows PowerShell
```

### Step 2: Configure System Settings

Edit `config.py` to customize settings:

```python
# Basic configuration (already provided in config.py)
# Modify these values as needed:

# Models (adjust based on your OpenAI plan)
SUMMARY_MODEL = "gpt-3.5-turbo"    # Faster, cheaper
JUDGE_MODEL = "gpt-4"              # More accurate, expensive
# JUDGE_MODEL = "gpt-3.5-turbo"   # Uncomment for budget option

# Vector storage settings
CHUNK_SIZE = 1000      # Reduce to 500 for memory-constrained systems
CHUNK_OVERLAP = 200    # Reduce to 100 for faster processing
SIMILARITY_SEARCH_K = 5  # Number of chunks to retrieve per query

# Processing settings  
BATCH_SIZE = 5         # Reduce to 3 for slower systems
MAX_SUMMARY_LENGTH = 300  # Adjust summary length
```

### Step 3: Create Data Directories

```bash
# Create required directories
mkdir sample_pdfs
mkdir results
mkdir logs

# Set permissions (macOS/Linux)
chmod 755 sample_pdfs results logs
```

---

## 🎯 First Run

### Step 1: Verify Configuration

```bash
# Run configuration verification
python verify_config.py
```

**Expected Output:**
```
🔧 VERIFYING CONFIG SETTINGS
===================================
✅ Config imported successfully

📋 Checking Original POC System Requirements:
   ✅ OPENAI_API_KEY: str
      API key set (length: 51)
   ✅ JUDGE_CRITERIA: dict
   ✅ N_CLUSTERS: int

🔍 Checking RAG System Requirements:
   ✅ RAG_EVALUATION_WEIGHTS: dict
   ✅ VECTOR_DB_PATH: str

📊 VERIFICATION SUMMARY:
✅ All config attributes present!
✅ Both systems should work correctly
```

### Step 2: Add Sample Documents

```bash
# Add 3-5 PDF documents to sample_pdfs/
# Any PDF documents work - research papers, business reports, manuals, etc.

# Examples of good document types:
# - Research papers from arXiv.org
# - Business reports 
# - Technical documentation
# - Policy documents
# - User manuals

# Check documents were added
ls sample_pdfs/
# Should show your PDF files
```

### Step 3: Quick Test Run

```bash
# Run quick validation test
python quick_simple_test.py
```

**Expected Output:**
```
🚀 QUICK SIMPLE RAG TEST
=========================
🔧 Setting up...
📊 Current documents: 0 chunks
📚 Processing PDFs...
🔄 Processing: document1.pdf
   ✅ Success: 5,234 characters extracted
📊 Successfully processed: 2/2 documents
✅ Loaded 2 documents

🧪 Testing 3 simple queries...

🔍 Query 1: What is this document about?
📝 Answer: This document discusses...
📚 From: document1.pdf
🏆 Score: 0.73

✅ Quick test complete!
```

---

## ✅ Validation & Testing

### Comprehensive System Test

```bash
# Test both systems together
python main.py --pdfs sample_pdfs/ --output results/ --mode both
```

### Individual System Tests

```bash
# Test original POC system only
python main.py --pdfs sample_pdfs/ --output results/ --mode summary

# Test RAG system only  
python main.py --pdfs sample_pdfs/ --output results/ --mode rag

# Comprehensive RAG testing with custom queries
python simple_test_system.py
```

### Expected Results

After successful runs, you should see:

**Files Created:**
```
results/
├── combined_report.html           # Executive summary
├── summary_evaluation_results.csv # Summary metrics (if mode=summary/both)
├── rag_evaluation_results.csv     # RAG metrics (if mode=rag/both)
└── summary_quality_categories.csv # Quality classifications
```

**Performance Indicators:**
- RAG scores between 0.6-0.85 indicate good performance
- Summary evaluation scores 6-9/10 indicate good quality
- Processing time: ~30 seconds per document for initial processing

---

## 🔧 Common Issues

### Issue 1: Python Version Problems

**Problem:** `python: command not found` or wrong version

**Solutions:**
```bash
# Try python3 instead of python
python3 --version

# Install Python (Ubuntu/Debian)
sudo apt install python3 python3-pip

# Install Python (macOS with Homebrew)
brew install python3

# Windows: Download from python.org
```

### Issue 2: OpenAI API Errors

**Problem:** `Invalid API key` or `Rate limit exceeded`

**Solutions:**
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Check API key format (should start with sk-)
# Check OpenAI account has credits
# Try using gpt-3.5-turbo instead of gpt-4 (cheaper)
```

**Budget-Friendly Config:**
```python
# In config.py - use cheaper models
SUMMARY_MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-3.5-turbo"  # Instead of gpt-4
```

### Issue 3: Memory Issues

**Problem:** `Out of memory` or very slow processing

**Solutions:**
```python
# In config.py - reduce resource usage
CHUNK_SIZE = 500        # Smaller chunks
SIMILARITY_SEARCH_K = 3 # Fewer retrieved chunks
BATCH_SIZE = 3          # Smaller batches
```

### Issue 4: PDF Processing Failures

**Problem:** No text extracted from PDFs

**Solutions:**
- Ensure PDFs contain extractable text (not scanned images)
- Try OCR tools for image-based PDFs
- Use different PDF files for testing

**Debug PDF Processing:**
```python
# Test PDF extraction manually
from simple_pdf_processor import SimplePDFProcessor
processor = SimplePDFProcessor()
doc = processor.process_single_pdf("sample_pdfs/test.pdf")
print(f"Extracted {len(doc['text'])} characters" if doc else "Failed")
```

### Issue 5: ChromaDB Issues

**Problem:** Vector database errors

**Solutions:**
```bash
# Clear and recreate vector database
rm -rf vector_db/
python quick_simple_test.py

# Check ChromaDB version compatibility
pip install --upgrade chromadb
```

---

## 🚀 Advanced Setup

### Docker Installation

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p sample_pdfs results vector_db

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "verify_config.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  llm-evaluator:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./sample_pdfs:/app/sample_pdfs
      - ./results:/app/results
      - ./vector_db:/app/vector_db
    command: python main.py --pdfs sample_pdfs --output results --mode both
```

### Production Deployment

```bash
# Environment-specific configs
cp config.py config_prod.py

# Production optimizations in config_prod.py:
BATCH_SIZE = 10              # Larger batches
SIMILARITY_SEARCH_K = 7      # More context
MAX_SUMMARY_LENGTH = 500     # Longer summaries

# Use production config
export CONFIG_FILE=config_prod.py
```

### Batch Processing Setup

```python
# batch_processor.py - Process multiple directories
import os
from pathlib import Path

def batch_process(directories):
    for directory in directories:
        output_dir = f"results_{Path(directory).name}"
        os.system(f"python main.py --pdfs {directory} --output {output_dir} --mode both")
        print(f"Completed: {directory} -> {output_dir}")

# Usage
directories = ["docs/quarterly_reports/", "docs/research_papers/", "docs/manuals/"]
batch_process(directories)
```

### Performance Monitoring

```python
# performance_monitor.py
import time
import psutil
import logging

logging.basicConfig(level=logging.INFO, filename='performance.log')

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        logging.info(f"Function: {func.__name__}")
        logging.info(f"Time: {end_time - start_time:.2f}s")
        logging.info(f"Memory: {end_memory - start_memory:.2f}MB")
        
        return result
    return wrapper

# Usage: Add @monitor_performance to functions
```

---

## 🎓 Next Steps

After successful setup:

1. **Customize Queries**: Edit `simple_test_system.py` with domain-specific questions
2. **Adjust Evaluation Criteria**: Modify weights in `config.py` for your use case  
3. **Scale Up**: Process larger document collections
4. **Integrate**: Use the system as part of larger workflows
5. **Monitor**: Track performance and adjust settings

## 📞 Getting Help

If you encounter issues:

1. **Run Diagnostics**: `python verify_config.py`
2. **Check Logs**: Look for error messages in console output
3. **Reduce Complexity**: Start with smaller document sets
4. **Test Components**: Use individual test scripts
5. **Check Resources**: Monitor memory and disk usage

**Support Checklist:**
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] OpenAI API key set correctly
- [ ] Sample PDFs added
- [ ] Configuration verified
- [ ] Quick test passed

---

**Setup Complete!** 🎉

Your LLM Evaluation System is now ready for use. Start with `python quick_simple_test.py` and then explore the full capabilities with `python main.py --mode both`.