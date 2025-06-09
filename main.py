#!/usr/bin/env python3
"""LLM Evaluation System - Main Entry Point"""

import argparse
import pandas as pd
import pickle
from pathlib import Path
from processor import PDFProcessor
from metrics_evals import MetricsEvaluator
from models import QualityModel
from vector_index import VectorQueryEngine
from report_generator import ReportGenerator
from html_generator import HTMLReportGenerator
import config

import warnings

warnings.filterwarnings("ignore")


def train_mode(args):
    """Train quality assessment model from PDFs"""
    print("🚀 Starting training mode...")
    print("📝 Generating multiple quality summaries per document for better training...")

    # Process PDFs and generate multiple summaries per document
    processor = PDFProcessor()
    documents = processor.process_directory(args.input, training_mode=True)

    print(f"✅ Generated {len(documents)} summaries from {len(documents) // 3} documents")

    # Evaluate summaries
    evaluator = MetricsEvaluator()
    metrics = evaluator.evaluate_batch(documents)

    # Train quality model
    model = QualityModel()
    model.fit(metrics)

    # Save model
    model_path = args.output / 'quality_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved quality model to {model_path}")

    # Generate training report
    html_gen = HTMLReportGenerator()
    html_gen.generate_training_report(metrics, args.output / 'training_report')


def inference_mode(args):
    """Create vector indexes from PDFs for inference"""
    print("🚀 Starting inference mode...")
    print("📚 Creating vector indexes from PDFs...")

    # Initialize vector engine
    engine = VectorQueryEngine()
    engine.index_pdfs(args.input)

    # Save the vector index
    engine.save_index(args.output / 'vector_index')

    # Get index statistics
    stats = engine.get_index_stats()

    print(f"\n✅ Vector index created successfully!")
    print(f"📊 Index Statistics:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Total documents: {stats['total_documents']}")
    print(f"   - Embedding dimensions: {stats['embedding_dimensions']}")

    # Save index statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(args.output / 'index_stats.csv', index=False)

    # Save chunk details for reference
    chunks_data = []
    for chunk in engine.chunks:
        chunks_data.append({
            'source': chunk['source'],
            'chunk_index': chunk['index'],
            'text_preview': chunk['text'][:100] + '...',
            'text_length': len(chunk['text'])
        })

    chunks_df = pd.DataFrame(chunks_data)
    chunks_df.to_csv(args.output / 'chunks_index.csv', index=False)

    # Generate index report
    html_gen = HTMLReportGenerator()
    html_gen.generate_index_report(stats, chunks_df, args.output / 'index_report')


def query_mode(args):
    """Answer questions using vector search"""
    print("🚀 Starting query mode...")

    # Load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Initialize vector engine
    engine = VectorQueryEngine()

    # Load existing index or create new one
    if args.index:
        print(f"📂 Loading vector index from {args.index}")
        engine.load_index(args.index)
    elif args.pdfs:
        print(f"📚 Creating vector index from {args.pdfs}")
        engine.index_pdfs(args.pdfs)
    else:
        print("❌ Error: Either --index or --pdfs must be provided for query mode")
        return

    # Process queries
    results = []

    # Determine source of questions
    if args.queries == "config":
        # Use questions from config.py
        questions = config.TEST_QUESTIONS
        print(f"📋 Using {len(questions)} questions from config.py")
    elif args.queries.endswith('.csv'):
        # Read from CSV file
        questions_df = pd.read_csv(args.queries)
        questions = questions_df['question'].tolist()
        print(f"📋 Loaded {len(questions)} questions from {args.queries}")
    else:
        # Single question provided
        questions = [args.queries]

    evaluator = MetricsEvaluator()

    for i, question in enumerate(questions, 1):
        print(f"\n📝 Processing [{i}/{len(questions)}]: {question}")

        # Get answer and chunks
        answer, chunks = engine.answer_question(question)

        # Evaluate answer quality
        eval_metrics = evaluator.evaluate_answer(question, answer, chunks)
        quality = model.predict_single(eval_metrics)

        # Clean chunks text for CSV (remove newlines, limit length)
        chunks_text = ' | '.join([chunk.replace('\n', ' ').replace('\r', ' ')[:200] for chunk in chunks])

        results.append({
            'question': question,
            'answer': answer.replace('\n', ' ').replace('\r', ' '),
            'chunks': chunks_text,
            'quality': quality,
            **eval_metrics
        })

    # Save results with proper escaping
    df = pd.DataFrame(results)
    df.to_csv(args.output / 'query_results.csv', index=False, escapechar='\\', quoting=1)

    # Generate report
    html_gen = HTMLReportGenerator()
    html_gen.generate_query_report(results, args.output / 'query_report')


def report_mode(args):
    """Generate professional reports from query results"""
    print("📊 Starting report generation mode...")

    # Check if input file exists
    if not args.input.exists():
        print(f"❌ Error: Input file {args.input} not found")
        return

    # Generate reports
    report_gen = ReportGenerator()
    report_gen.generate_reports(args.input, args.output)

    print(f"\n✅ Reports generated successfully in {args.output}/")
    print("📄 Files created:")
    print(f"   - professional_report.html (Executive summary)")
    print(f"   - metrics_summary.csv (Aggregated metrics)")
    print(f"   - quality_breakdown.csv (Quality distribution)")
    print(f"   - question_performance.csv (Per-question analysis)")
    print(f"   - detailed_metrics.csv (All metrics for each question)")


def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation System')
    parser.add_argument('mode', choices=['train', 'inference', 'query', 'report'],
                        help='Operation mode')
    parser.add_argument('--input', type=Path, help='Input directory or file')
    parser.add_argument('--output', type=Path, default=Path('output'),
                        help='Output directory')
    parser.add_argument('--model', type=Path, help='Model file for train/query mode')
    parser.add_argument('--pdfs', type=Path, help='PDFs for query mode (creates new index)')
    parser.add_argument('--index', type=Path, help='Pre-built vector index for query mode')
    parser.add_argument('--queries', help='Query file or single question')

    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)

    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'inference':
        inference_mode(args)
    elif args.mode == 'query':
        if not args.model:
            print("❌ Error: --model is required for query mode")
            return
        if not (args.index or args.pdfs):
            print("❌ Error: Either --index or --pdfs is required for query mode")
            return
        query_mode(args)
    elif args.mode == 'report':
        if not args.input:
            print("❌ Error: --input is required for report mode")
            return
        report_mode(args)


if __name__ == "__main__":
    main()