#!/usr/bin/env python3
"""LLM Evaluation System - Main Entry Point"""

import argparse
import os
import pandas as pd
import pickle
from pathlib import Path
from processor import PDFProcessor
from metrics_evals import MetricsEvaluator
from models import QualityModel
from vector_index import VectorQueryEngine
import config

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    generate_report(metrics, args.output / 'training_report')


def inference_mode(args):
    """Process new PDFs and evaluate quality"""
    print("🚀 Starting inference mode...")

    # Load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Process PDFs (single high-quality summary per document)
    processor = PDFProcessor()
    documents = processor.process_directory(args.input, training_mode=False)

    # Evaluate and predict quality
    evaluator = MetricsEvaluator()
    metrics = evaluator.evaluate_batch(documents)
    results = model.predict(metrics)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output / 'inference_results.csv', index=False)

    # Generate report
    generate_report(results, args.output / 'inference_report')


def query_mode(args):
    """Answer questions using vector search"""
    print("🚀 Starting query mode...")

    # Load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Initialize vector engine
    engine = VectorQueryEngine()
    engine.index_pdfs(args.pdfs)

    # Process queries
    results = []
    questions = pd.read_csv(args.queries) if args.queries.endswith('.csv') else [{'question': args.queries}]

    evaluator = MetricsEvaluator()

    for item in questions:
        question = item['question'] if isinstance(item, dict) else item
        print(f"\n📝 Processing: {question}")

        # Get answer and chunks
        answer, chunks = engine.answer_question(question)

        # Evaluate answer quality
        eval_metrics = evaluator.evaluate_answer(question, answer, chunks)
        quality = model.predict_single(eval_metrics)

        results.append({
            'question': question,
            'answer': answer,
            'chunks': ' | '.join(chunks),
            'quality': quality,
            **eval_metrics
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output / 'query_results.csv', index=False)

    # Generate report
    generate_query_report(results, args.output / 'query_report')


def generate_report(data, output_path):
    """Generate HTML report"""
    df = pd.DataFrame(data)

    # Quality distribution
    quality_dist = df['quality'].value_counts() if 'quality' in df else {}

    # Check if this is training data with intended quality
    intended_quality_section = ""
    if 'intended_quality' in df.columns:
        intended_dist = df['intended_quality'].value_counts()
        intended_quality_section = f"""
        <h2>Training Data Distribution</h2>
        <p><strong>Summaries per Quality Level:</strong></p>
        <ul>
            {"".join([f'<li><strong>{k}:</strong> {v} summaries</li>' for k, v in intended_dist.items()])}
        </ul>

        <h2>Model Performance</h2>
        <p>How well the model clusters align with intended quality:</p>
        <table>
            <tr><th>Predicted</th><th>High</th><th>Medium</th><th>Low</th></tr>
            {generate_confusion_matrix(df)}
        </table>
        """

    html = f"""
    <html>
    <head>
        <title>LLM Evaluation Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ color: #2196F3; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>📊 LLM Evaluation Report</h1>

        <h2>Summary Statistics</h2>
        <p><strong>Total Documents:</strong> {len(df) if 'intended_quality' not in df else len(df) // 3}</p>
        <p><strong>Total Summaries:</strong> {len(df)}</p>
        <p><strong>Average Score:</strong> {df['overall_score'].mean():.2f}/10</p>

        {intended_quality_section}

        <h2>Quality Distribution (Model Predictions)</h2>
        <ul>
            {"".join([f'<li><strong>{k}:</strong> {v} summaries</li>' for k, v in quality_dist.items()])}
        </ul>

        <h2>Top Performing Summaries</h2>
        <table>
            <tr><th>Document</th><th>Quality</th><th>Score</th></tr>
            {generate_table_rows(df.nlargest(5, 'overall_score')[['document', 'quality', 'overall_score']])}
        </table>
    </body>
    </html>
    """

    with open(f"{output_path}.html", 'w') as f:
        f.write(html)


def generate_confusion_matrix(df):
    """Generate confusion matrix HTML for training report"""
    if 'intended_quality' not in df.columns:
        return ""

    # Create confusion matrix
    matrix = {}
    for intended in ['high', 'medium', 'low']:
        matrix[intended] = {}
        for predicted in ['High', 'Medium', 'Low']:
            count = len(df[(df['intended_quality'] == intended) & (df['quality'] == predicted)])
            matrix[intended][predicted] = count

    rows = []
    for predicted in ['High', 'Medium', 'Low']:
        row = f"<tr><td><strong>{predicted}</strong></td>"
        for intended in ['high', 'medium', 'low']:
            count = matrix[intended][predicted]
            total = df[df['intended_quality'] == intended].shape[0]
            pct = (count / total * 100) if total > 0 else 0
            row += f"<td>{count} ({pct:.0f}%)</td>"
        row += "</tr>"
        rows.append(row)

    return "".join(rows)


def generate_query_report(results, output_path):
    """Generate query-specific HTML report"""
    df = pd.DataFrame(results)

    html = f"""
    <html>
    <head>
        <title>Query Evaluation Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .question {{ background: #e3f2fd; padding: 10px; margin: 10px 0; }}
            .answer {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
            .quality-High {{ color: green; }}
            .quality-Medium {{ color: orange; }}
            .quality-Low {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>📝 Query Evaluation Report</h1>
        <p><strong>Total Questions:</strong> {len(df)}</p>

        {"".join([f'''
        <div class="question">
            <h3>Q: {row['question']}</h3>
            <div class="answer">
                <p><strong>Answer:</strong> {row['answer'][:500]}...</p>
                <p><strong>Quality:</strong> <span class="quality-{row['quality']}">{row['quality']}</span></p>
                <p><strong>Relevance:</strong> {row.get('relevance_score', 0):.2f} | 
                   <strong>Coherence:</strong> {row.get('coherence_score', 0):.2f}</p>
            </div>
        </div>
        ''' for _, row in df.iterrows()])}
    </body>
    </html>
    """

    with open(f"{output_path}.html", 'w') as f:
        f.write(html)


def generate_table_rows(df):
    """Helper to generate HTML table rows"""
    return "".join([f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]:.2f}</td></tr>"
                    for row in df.values])


def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation System')
    parser.add_argument('mode', choices=['train', 'inference', 'query'],
                        help='Operation mode')
    parser.add_argument('--input', type=Path, help='Input directory')
    parser.add_argument('--output', type=Path, default=Path('output'),
                        help='Output directory')
    parser.add_argument('--model', type=Path, help='Model file for inference/query')
    parser.add_argument('--pdfs', type=Path, help='PDFs for query mode')
    parser.add_argument('--queries', help='Query file or single question')

    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)

    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'inference':
        inference_mode(args)
    elif args.mode == 'query':
        query_mode(args)


if __name__ == "__main__":
    main()