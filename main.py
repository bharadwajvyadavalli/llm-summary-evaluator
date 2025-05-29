#!/usr/bin/env python3
"""
LLM Evaluation Service POC - Main Script (Updated)
Now works with both summary evaluation and RAG Q&A evaluation
"""

import argparse
import os
import pandas as pd
from simple_pdf_processor import SimplePDFProcessor  # Updated import
from mathematical_metrics import MathematicalMetrics
from llm_judge import LLMJudge
from clustering_model import ClusteringModel
import config


def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Service POC')
    parser.add_argument('--pdfs', required=True, help='Directory with PDF files')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--mode', default='summary', choices=['summary', 'rag', 'both'],
                        help='Evaluation mode: summary (original), rag (new), or both')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("🚀 Starting LLM Evaluation System...")
    print(f"📋 Mode: {args.mode}")

    # Step 1: Process PDFs (now works with any documents)
    print("📄 Processing PDF documents...")
    processor = SimplePDFProcessor()  # Updated processor
    documents = processor.process_pdfs(args.pdfs)
    print(f"✅ Processed {len(documents)} documents")

    if not documents:
        print("❌ No documents to evaluate")
        return

    results = {}

    # Summary Evaluation (Original POC)
    if args.mode in ['summary', 'both']:
        print("\n📊 Running Summary Evaluation (Original POC)...")

        # For summary evaluation, we need abstracts
        # Filter documents that have meaningful summaries
        summary_docs = []
        for doc in documents:
            if doc.get('summary') and len(doc['summary']) > 50:
                # Create abstract from summary for evaluation
                doc['abstract'] = doc['summary']  # Use summary as reference
                summary_docs.append(doc)

        if summary_docs:
            print(f"📝 Evaluating summaries for {len(summary_docs)} documents...")

            # Step 2: Calculate mathematical metrics
            print("🔢 Calculating mathematical metrics...")
            math_metrics = MathematicalMetrics()
            math_results = math_metrics.evaluate_batch(summary_docs)

            # Step 3: LLM Judge evaluation
            print("🤖 Running LLM judge evaluation...")
            judge = LLMJudge()
            judge_results = judge.evaluate_batch(summary_docs)

            # Step 4: Combine metrics
            print("📊 Combining all metrics...")
            combined_df = combine_metrics(math_results, judge_results)

            # Step 5: Quality categorization
            print("🎯 Categorizing quality (Low/Medium/High)...")
            clustering = ClusteringModel()
            summary_results = clustering.categorize_quality(combined_df)

            results['summary_evaluation'] = summary_results

        else:
            print("⚠️ No documents with adequate content for summary evaluation")

    # RAG Evaluation (New System)
    if args.mode in ['rag', 'both']:
        print("\n🔍 Running RAG Q&A Evaluation (New System)...")

        from vector_store_manager import VectorStoreManager
        from simple_rag_agent import SimpleRAGAgent
        from rag_evaluator import RAGEvaluator

        # Initialize RAG system
        vector_manager = VectorStoreManager()
        vector_manager.add_documents(documents)
        rag_agent = SimpleRAGAgent(vector_manager)
        rag_evaluator = RAGEvaluator()

        # Test queries for RAG evaluation
        test_queries = [
            "What are the main topics covered in these documents?",
            "What key information is provided?",
            "What are the important details mentioned?",
            "What conclusions or recommendations are made?",
            "What specific facts or data are presented?"
        ]

        print(f"🧪 Testing {len(test_queries)} queries...")

        rag_results = []
        for query in test_queries:
            try:
                response = rag_agent.query(query, k=3)
                metrics = rag_evaluator.evaluate_response(response)

                rag_results.append({
                    'query': query,
                    'response': response.response,
                    'sources': ', '.join(response.sources),
                    **metrics
                })
            except Exception as e:
                print(f"⚠️ Query failed: {e}")

        if rag_results:
            rag_df = pd.DataFrame(rag_results)
            results['rag_evaluation'] = rag_df

    # Step 6: Save results
    print("💾 Saving results...")
    save_results(results, args.output, args.mode)

    print("\n🎉 Evaluation Complete!")
    print(f"📊 Results saved to: {args.output}/")

    # Print summary
    if 'summary_evaluation' in results:
        quality_counts = results['summary_evaluation']['quality_category'].value_counts()
        print(f"\n📈 Summary Quality Distribution:")
        for category, count in quality_counts.items():
            print(f"   • {category}: {count} documents")

    if 'rag_evaluation' in results:
        avg_rag_score = results['rag_evaluation']['rag_score'].mean()
        print(f"\n🎯 RAG Performance:")
        print(f"   • Average RAG Score: {avg_rag_score:.3f}")
        print(f"   • Queries tested: {len(results['rag_evaluation'])}")


def combine_metrics(math_results, judge_results):
    """Combine mathematical and judge metrics"""
    # Convert to DataFrames and merge
    math_df = pd.DataFrame(math_results)
    judge_df = pd.DataFrame(judge_results)

    combined = pd.concat([math_df, judge_df], axis=1)
    return combined


def save_results(results, output_dir, mode):
    """Save evaluation results"""

    if 'summary_evaluation' in results:
        # Save summary evaluation results
        summary_df = results['summary_evaluation']
        summary_df.to_csv(f"{output_dir}/summary_evaluation_results.csv", index=False)

        # Save quality summary
        quality_summary = summary_df[['document_name', 'quality_category', 'overall_score']].copy()
        quality_summary.to_csv(f"{output_dir}/summary_quality_categories.csv", index=False)

    if 'rag_evaluation' in results:
        # Save RAG evaluation results
        rag_df = results['rag_evaluation']
        rag_df.to_csv(f"{output_dir}/rag_evaluation_results.csv", index=False)

    # Generate combined HTML report
    generate_html_report(results, output_dir, mode)


def generate_html_report(results, output_dir, mode):
    """Generate combined HTML summary report"""

    html_parts = [
        "<html>",
        "<head><title>LLM Evaluation Results</title></head>",
        "<body style='font-family: Arial; margin: 20px;'>",
        "<h1>📊 LLM Evaluation System Results</h1>",
        f"<p><strong>Evaluation Mode:</strong> {mode}</p>"
    ]

    if 'summary_evaluation' in results:
        summary_df = results['summary_evaluation']
        quality_counts = summary_df['quality_category'].value_counts()
        avg_score = summary_df['overall_score'].mean()

        html_parts.extend([
            "<h2>📝 Summary Evaluation Results</h2>",
            "<h3>Quality Distribution</h3>",
            "<ul>"
        ])

        for cat, count in quality_counts.items():
            html_parts.append(f'<li><strong>{cat}:</strong> {count} documents</li>')

        html_parts.extend([
            "</ul>",
            f"<p><strong>Average Quality Score:</strong> {avg_score:.2f}/10</p>"
        ])

    if 'rag_evaluation' in results:
        rag_df = results['rag_evaluation']
        avg_rag_score = rag_df['rag_score'].mean()

        html_parts.extend([
            "<h2>🔍 RAG Q&A Evaluation Results</h2>",
            f"<p><strong>Queries Tested:</strong> {len(rag_df)}</p>",
            f"<p><strong>Average RAG Score:</strong> {avg_rag_score:.3f}</p>",
            "<p><strong>Top Performing Query:</strong></p>",
            f"<p><em>{rag_df.loc[rag_df['rag_score'].idxmax(), 'query']}</em></p>"
        ])

    html_parts.extend([
        "<h2>📁 Generated Files</h2>",
        "<ul>"
    ])

    if 'summary_evaluation' in results:
        html_parts.extend([
            "<li>summary_evaluation_results.csv - Detailed summary metrics</li>",
            "<li>summary_quality_categories.csv - Summary classifications</li>"
        ])

    if 'rag_evaluation' in results:
        html_parts.append("<li>rag_evaluation_results.csv - RAG Q&A metrics</li>")

    html_parts.extend([
        "</ul>",
        "</body>",
        "</html>"
    ])

    with open(f"{output_dir}/combined_report.html", 'w') as f:
        f.write('\n'.join(html_parts))


if __name__ == "__main__":
    main()