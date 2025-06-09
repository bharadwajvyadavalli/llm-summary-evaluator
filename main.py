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
    generate_index_report(stats, chunks_df, args.output / 'index_report')


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


def generate_index_report(stats, chunks_df, output_path):
    """Generate HTML report for vector index"""
    # Group chunks by source
    source_stats = chunks_df.groupby('source').agg({
        'chunk_index': 'count',
        'text_length': ['mean', 'sum']
    }).round(2)

    html = f"""
    <html>
    <head>
        <title>Vector Index Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ color: #2196F3; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🔍 Vector Index Report</h1>

        <h2>Index Overview</h2>
        <ul>
            <li><strong>Total Documents:</strong> {stats['total_documents']}</li>
            <li><strong>Total Chunks:</strong> {stats['total_chunks']}</li>
            <li><strong>Embedding Dimensions:</strong> {stats['embedding_dimensions']}</li>
            <li><strong>Average Chunks per Document:</strong> {stats['total_chunks'] / stats['total_documents']:.1f}</li>
        </ul>

        <h2>Document Statistics</h2>
        <table>
            <tr>
                <th>Document</th>
                <th>Number of Chunks</th>
                <th>Avg Chunk Length</th>
                <th>Total Characters</th>
            </tr>
            {"".join([f'''
            <tr>
                <td>{doc}</td>
                <td>{int(source_stats.loc[doc, ('chunk_index', 'count')])}</td>
                <td>{source_stats.loc[doc, ('text_length', 'mean')]:.0f}</td>
                <td>{source_stats.loc[doc, ('text_length', 'sum')]:.0f}</td>
            </tr>
            ''' for doc in source_stats.index])}
        </table>

        <h2>Ready for Queries</h2>
        <p>The vector index is now ready to answer questions about the indexed documents.</p>
        <p>You can use query mode to ask questions about these documents.</p>
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


if __name__ == "__main__":
    main()