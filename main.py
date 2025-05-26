#!/usr/bin/env python3
"""
LLM Evaluation Service POC - Main Script
"""

import argparse
import os
import pandas as pd
from document_processor import DocumentProcessor
from mathematical_metrics import MathematicalMetrics
from llm_judge import LLMJudge
from clustering_model import ClusteringModel
import config

def main():
    parser = argparse.ArgumentParser(description='LLM Evaluation Service POC')
    parser.add_argument('--pdfs', required=True, help='Directory with PDF files')
    parser.add_argument('--output', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("🚀 Starting LLM Summary Evaluation...")
    
    # Step 1: Process PDFs
    print("📄 Processing PDF documents...")
    processor = DocumentProcessor()
    documents = processor.process_pdfs(args.pdfs)
    print(f"✅ Processed {len(documents)} documents")
    
    # Step 2: Calculate mathematical metrics
    print("🔢 Calculating mathematical metrics...")
    math_metrics = MathematicalMetrics()
    math_results = math_metrics.evaluate_batch(documents)
    
    # Step 3: LLM Judge evaluation  
    print("🤖 Running LLM judge evaluation...")
    judge = LLMJudge()
    judge_results = judge.evaluate_batch(documents)
    
    # Step 4: Combine metrics
    print("📊 Combining all metrics...")
    combined_df = combine_metrics(math_results, judge_results)
    
    # Step 5: Quality categorization
    print("🎯 Categorizing quality (Low/Medium/High)...")
    clustering = ClusteringModel()
    final_results = clustering.categorize_quality(combined_df)
    
    # Step 6: Save results
    print("💾 Saving results...")
    save_results(final_results, args.output)
    
    print("\n🎉 Evaluation Complete!")
    print(f"📊 Results saved to: {args.output}/")
    
    # Print summary
    quality_counts = final_results['quality_category'].value_counts()
    print(f"\n📈 Quality Distribution:")
    for category, count in quality_counts.items():
        print(f"   • {category}: {count} documents")

def combine_metrics(math_results, judge_results):
    """Combine mathematical and judge metrics"""
    # Convert to DataFrames and merge
    math_df = pd.DataFrame(math_results)
    judge_df = pd.DataFrame(judge_results)
    
    combined = pd.concat([math_df, judge_df], axis=1)
    return combined

def save_results(results_df, output_dir):
    """Save evaluation results"""
    # Save detailed results
    results_df.to_csv(f"{output_dir}/evaluation_results.csv", index=False)
    
    # Save quality summary
    quality_summary = results_df[['document_name', 'quality_category', 'overall_score']].copy()
    quality_summary.to_csv(f"{output_dir}/quality_categories.csv", index=False)
    
    # Generate simple HTML report
    generate_html_report(results_df, output_dir)

def generate_html_report(results_df, output_dir):
    """Generate simple HTML summary report"""
    quality_counts = results_df['quality_category'].value_counts()
    avg_score = results_df['overall_score'].mean()
    
    html = f"""
    <html>
    <head><title>LLM Evaluation Results</title></head>
    <body style="font-family: Arial; margin: 20px;">
        <h1>📊 LLM Summary Evaluation Results</h1>
        
        <h2>Quality Distribution</h2>
        <ul>
            {chr(10).join([f'<li><strong>{cat}:</strong> {count} documents</li>' for cat, count in quality_counts.items()])}
        </ul>
        
        <h2>Overall Statistics</h2>
        <p><strong>Total Documents:</strong> {len(results_df)}</p>
        <p><strong>Average Quality Score:</strong> {avg_score:.2f}/10</p>
        
        <h2>Files Generated</h2>
        <ul>
            <li>evaluation_results.csv - Detailed metrics</li>
            <li>quality_categories.csv - Final classifications</li>
        </ul>
    </body>
    </html>
    """
    
    with open(f"{output_dir}/summary_report.html", 'w') as f:
        f.write(html)

if __name__ == "__main__":
    main()